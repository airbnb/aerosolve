package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core._
import com.airbnb.aerosolve.core.features.SparseLabeledPoint
import com.airbnb.aerosolve.core.function._
import com.airbnb.aerosolve.core.models.AdditiveModel
import com.airbnb.aerosolve.core.transforms.Transformer
import com.airbnb.aerosolve.training.pipeline.NDTreePipeline.{FeatureStats, NDTreePipelineParams}
import com.airbnb.aerosolve.training.pipeline.{NDTreePipeline, PipelineUtil}
import com.typesafe.config.Config
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Accumulator, SparkContext}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.language.implicitConversions
import scala.util.{Random, Try}

/**
  * Additive Model Trainer
  * By default, we use a spline function to represent a float feature; use linear function to represent a string feature.
  * Additionally, float features that are specified as 'linear_feature' in config are also represented by a linear function.
  *
  * Model is fitted using [[https://en.wikipedia.org/wiki/Backfitting_algorithm Backfitting Algorithm]] where weight vectors
  * for each feature is updated independently using SGD and after each iteration spline features are passed under a smoothing
  * operator. This is a simplified implementation of GAM where features are bucketed into exactly where knots are with some
  * flexibility being provided by applying multiscaling to the bucketing scheme.
  */
//noinspection NameBooleanParameters
object AdditiveModelTrainer {
  private final val log: Logger = LoggerFactory.getLogger("AdditiveModelTrainer")
  private val random = new Random()

  object LossFunctions extends Enumeration {
    /**
      * Adjust learning rate according to class sampling weights
      */
    def getLearningRate(label: Double, params: AdditiveTrainerParams, learningRate: Double): Float = {
      params.classWeights(label.toInt) * learningRate.toFloat
    }

    abstract class LossFunction extends super.Val {
      def update(model: AdditiveModel,
                 point: SparseLabeledPoint,
                 params: AdditiveTrainerParams,
                 learningRate: Double,
                 lossOnly: Boolean): Double
    }

    implicit def valueToPlanetVal(x: Value): LossFunction = x.asInstanceOf[LossFunction]

    val LOGISTIC = new LossFunction {
      // http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
      // We rescale by 1 / p so that at inference time we don't have to scale by p.
      // In our case p = 1.0 - dropout rate
      def update(model: AdditiveModel,
                 point: SparseLabeledPoint,
                 params: AdditiveTrainerParams,
                 learningRate: Double,
                 lossOnly: Boolean): Double = {
        val seed = random.nextLong()
        val rand = new Random(seed)

        val prediction = model.scoreFeatures(point, if (lossOnly) 0 else params.dropout, rand.self)
        val label = point.label

        // To prevent blowup.
        val corr = scala.math.min(10.0, label * prediction)
        val expCorr = scala.math.exp(corr)
        val loss = scala.math.log(1.0 + 1.0 / expCorr)

        if (!lossOnly) {
          val grad = -label / (1.0 + expCorr)
          val gradWithLearningRate = (grad * getLearningRate(label, params, learningRate)).toFloat

          rand.setSeed(seed)
          model.update(gradWithLearningRate, point, params.dropout, rand.self)
        }

        loss
      }
    }

    val HINGE = new LossFunction {
      def update(model: AdditiveModel,
                 point: SparseLabeledPoint,
                 params: AdditiveTrainerParams,
                 learningRate: Double,
                 lossOnly: Boolean): Double = {
        val seed = random.nextLong()
        val rand = new Random(seed)

        val prediction = model.scoreFeatures(point, if (lossOnly) 0 else params.dropout, rand.self)
        val label = point.label

        val loss = scala.math.max(0.0, params.margin - label * prediction)
        if (loss > 0.0 && !lossOnly) {
          val gradWithLearningRate = -label.toFloat * getLearningRate(label, params, learningRate)

          rand.setSeed(seed)
          model.update(gradWithLearningRate, point, params.dropout, rand.self)
        }

        loss
      }
    }

    val REGRESSION = new LossFunction {
      def update(model: AdditiveModel,
                 point: SparseLabeledPoint,
                 params: AdditiveTrainerParams,
                 learningRate: Double,
                 lossOnly: Boolean): Double = {
        val seed = random.nextLong()
        val rand = new Random(seed)

        val prediction = model.scoreFeatures(point, if (lossOnly) 0 else params.dropout, rand.self)
        val label = point.label

        val loss = prediction - label
        if (loss > params.epsilon) {
          rand.setSeed(seed)
          model.update(learningRate.toFloat, point, params.dropout, rand.self)
        } else if (loss < -params.epsilon) {
          rand.setSeed(seed)
          model.update(-learningRate.toFloat, point, params.dropout, rand.self)
        }

        // absolute difference
        math.abs(loss)
      }
    }
  }

  import LossFunctions._

  case class SgdParams(params: AdditiveTrainerParams,
                       exampleCount: Accumulator[Long],
                       loss: Accumulator[Double])

  case class LossParams(function: LossFunction,
                        lossMod: Int,
                        // stop iterations early if validation loss has not improved for this number of iterations
                        earlyStopping: Int,
                        convergenceTolerance: Double,
                        // default to None, if set data is split into training and validation accordingly
                        // during each iteration. Loss is computed without updating the additive model itself
                        // decay_by_validation_loss can use this loss to guide the decay of learning rate
                        isTraining: Option[Example => Boolean])

  case class InitParams(initModelPath: String,
                        // default to false, if set to true,
                        // it use initModelPath's model's function
                        // without recomputing functions from data.
                        onlyUseInitModelFunctions: Boolean,
                        linearFeatureFamilies: Set[String],
                        priors: Array[String],
                        minCount: Int,
                        nDTreePipelineParams: NDTreePipelineParams,
                        initQuantiles: Seq[Double],
                        initSubsample: Double)

  case class LearningRateParams(initialLearningRate: Double,
                                learningRateDecay: Double,
                                minLearningRate: Double,
                                decayByValidationLoss: Boolean,
                                decayLookbackIterations: Int)

  case class IOParams(modelOutput: String,
                      // default to true, if set to false, no shuffling for each iteration
                      shuffle: Boolean,
                      // if checkPointDir set, transformed training data will be saved
                      // to hdfs checkPointDir, to avoid rerun if training failed.
                      checkPointDir: String,
                      // cache transformed training data when starting
                      // this is useful if subsample * iteration >> 1 (i.e. data are reused throughout the iterations)
                      storageLevel: StorageLevel,
                      saveModelEachIteration: Boolean)

  case class AdditiveTrainerParams(numBins: Int,
                                   numBags: Int,
                                   rankKey: String,
                                   loss: LossParams,
                                   learningRate: LearningRateParams,
                                   dropout: Double,
                                   subsample: Double,
                                   // defaults to subsample. this is only used when storage_level is NONE (not set)
                                   validationSubsample: Double,
                                   partitionSubsample: Double,
                                   margin: Double,
                                   multiscale: Array[Int],
                                   smoothingTolerance: Double,
                                   smoothingByPercentage: Boolean,
                                   linfinityThreshold: Double,
                                   linfinityCap: Double,
                                   capIterations: Int,
                                   threshold: Double,
                                   epsilon: Double, // epsilon used in epsilon-insensitive loss for regression training
                                   init: InitParams,
                                   classWeights: Map[Int, Float],
                                   io: IOParams)

  def train(sc: SparkContext, input: RDD[Example], config: Config, key: String): AdditiveModel =
    train(sc, (frac: Double) => input.sample(false, frac), config, key)

  // save init model to avoid recompute dynamic bucketing
  def saveInitModel(sc: SparkContext,
                    input: Double => RDD[Example],
                    config: Config,
                    key: String,
                    isTraining: Option[Example => Boolean] = None): Unit = {
    val trainConfig = config.getConfig(key)
    val params = loadTrainingParameters(trainConfig, isTraining)
    val transformer = new Transformer(config, key)
    val transformed = (frac: Double) => LinearRankerUtils.makePointwiseFloat(input(frac), transformer)
    val model = modelInitialization(sc, transformed, params)
    val output = config.getString(key + ".init_model")
    TrainingUtils.saveModel(model, output)
  }

  def train(sc: SparkContext,
            input: Double => RDD[Example],
            config: Config,
            key: String,
            isTraining: Option[Example => Boolean] = None): AdditiveModel = {
    val trainConfig = config.getConfig(key)
    val iterations: Int = trainConfig.getInt("iterations")

    val params = loadTrainingParameters(trainConfig, isTraining)
    log.info(s"Training using ${params.loss}, ${params.learningRate}")

    // sample before we transform as it can be very expensive especially for crossing
    // NB: this assumes we don't add/remove observations during transformation
    val transformer = new Transformer(config, key)

    // transform example as-is to initialize model which we can base our vector compression upon
    val transformed = (frac: Double) => LinearRankerUtils.makePointwiseFloat(input(frac), transformer)
    var model = modelInitialization(sc, transformed, params)
    var modelBC = sc.broadcast(model.generateFeatureIndexer())
    var bestModel = model

    // broadcast a separate copy of model because the modelBC will be destroyed after each iteration
    val Array(trainVectorRDD, validationVectorRDD) = transformFeatureVectorAndSplit(sc, input, params, transformer, sc.broadcast(model))

    var learningRate = params.learningRate.initialLearningRate
    var trainingLosses = List(Double.MaxValue)
    var validationLosses = List(Double.MaxValue)
    var bestValidationLoss = Double.MaxValue
    var bestIteration = 0
    val sgdParams = SgdParams(params, sc.accumulator(0), sc.accumulator(0))
    var i = 0
    while (i < iterations &&
      (params.loss.earlyStopping == 0 || (i - bestIteration) < params.loss.earlyStopping) &&
      (trainingLosses.length < 2 || params.loss.convergenceTolerance <= 0 || (trainingLosses(1) - trainingLosses(0) > params.loss.convergenceTolerance))
    ) {
      i += 1

      log.info(s"Iteration $i")
      model = sgdTrain(trainVectorRDD, sgdParams, modelBC, learningRate)
      // discard broadcast model that is now obsolete
      modelBC.destroy()

      if ((i % params.capIterations) == 0) {
        // don't drop functions yet to keep vector representation consistent
        deleteSmallFunctions(model, params.linfinityThreshold, false)
      }

      val loss = sgdParams.loss.value / sgdParams.exampleCount.value
      trainingLosses ::= loss
      log.info(s"iterations $i Loss = $loss count = ${sgdParams.exampleCount.value} lr = $learningRate")

      // reset loss accumulator
      sgdParams.loss.setValue(0.0)
      sgdParams.exampleCount.setValue(0L)

      // validate loss on validation dataset
      modelBC = sc.broadcast(model)
      sgdTrain(validationVectorRDD, sgdParams, modelBC, learningRate, true)
      if (sgdParams.exampleCount.value > 0) {
        val validationLoss = sgdParams.loss.value / sgdParams.exampleCount.value
        log.info(s"iterations $i Loss = $validationLoss count = ${sgdParams.exampleCount.value} (validation)")

        // reset loss accumulator
        sgdParams.loss.setValue(0.0)
        sgdParams.exampleCount.setValue(0L)

        // update validation loss and save model if we get a better result
        if (validationLoss < bestValidationLoss) {
          bestValidationLoss = validationLoss
          bestIteration = i

          bestModel = model.clone()
          // drop zero functions before saving
          deleteSmallFunctions(bestModel, params.linfinityThreshold, true)
          val savePath = if (params.io.saveModelEachIteration) {
            s"${params.io.modelOutput}_$i"
          } else {
            params.io.modelOutput
          }
          TrainingUtils.saveModel(bestModel, savePath)
        }

        if (!params.learningRate.decayByValidationLoss ||
          // decay learning rate only when it hasn't decreased in last `x` iterations
          (validationLoss > validationLosses.take(params.learningRate.decayLookbackIterations).max)) {
          learningRate = math.max(params.learningRate.minLearningRate, learningRate * params.learningRate.learningRateDecay)
        }

        validationLosses ::= validationLoss
      } else {
        bestModel = model.clone()
        if (params.io.saveModelEachIteration) {
          // drop zero functions before saving
          deleteSmallFunctions(bestModel, params.linfinityThreshold, true)
          TrainingUtils.saveModel(bestModel, s"${params.io.modelOutput}_$i")
        }

        learningRate = math.max(params.learningRate.minLearningRate, learningRate * params.learningRate.learningRateDecay)
      }
    }
    // now we can actually drop all those functions
    deleteSmallFunctions(bestModel, params.linfinityThreshold, true)
    bestModel
  }

  /**
    * Transform input data according to transformer and map them to [[SparseLabeledPoint]] according to model for
    * space efficiency. Input data is split into Array(Training Data, Validation Data) if validation data split
    * has been set. Data is persisted accordingly. See inline comments w.r.t specific optimization taken
    * for each scenario.
    */
  def transformFeatureVectorAndSplit(sc: SparkContext,
                                     input: (Double) => RDD[Example],
                                     params: AdditiveTrainerParams,
                                     transformer: Transformer,
                                     modelBC: Broadcast[AdditiveModel]): Array[(Double) => RDD[SparseLabeledPoint]] = {
    val isTraining = params.loss.isTraining

    if (params.io.storageLevel == StorageLevel.NONE) {
      if (isTraining.isDefined) {
        // we will cache the validation set even if no persistence is requested for the training data because it
        // is likely to be very small and requires no shuffling
        // NB: we also apply sub-sampling on validation data for consistency
        // NB: sub-sampling is stationary across iterations for efficiency
        // if we need validation data, we must split on the full dataset to avoid contamination
        val validationRDD = input(params.validationSubsample).filter(!isTraining.get(_))
        val validationVectorRDD = LinearRankerUtils.makePointwiseFloatVector(validationRDD, transformer, params, modelBC, _ => false)
          // reduce number of partition to reduce reporting overhead
          .coalesce(1000)
          .cache()

        Array(
          (frac: Double) => LinearRankerUtils.makePointwiseFloatVector(input(frac).filter(isTraining.get), transformer, params, modelBC),
          (frac: Double) => validationVectorRDD
        )
      } else {
        Array(
          (frac: Double) => LinearRankerUtils.makePointwiseFloatVector(input(frac), transformer, params, modelBC),
          (frac: Double) => sc.emptyRDD[SparseLabeledPoint]
        )
      }
    } else {
      // we persist the entire dataset first then run sampling during training
      val transformedVectorRDD =
      LinearRankerUtils.makePointwiseFloatVector(input(1.0), transformer, params, modelBC, isTraining.getOrElse(_ => true))
        .persist(params.io.storageLevel)

      (if (isTraining.isDefined) {
        Array(
          transformedVectorRDD.filter(_.isTraining),
          transformedVectorRDD.filter(!_.isTraining)
        )
      } else {
        Array(transformedVectorRDD, sc.emptyRDD[SparseLabeledPoint])
      })
        .map { rdd => (frac: Double) => rdd.sample(false, frac) }
    }
  }

  /**
    * During each iteration, we:
    *
    * 1. Sample dataset with subsample (this is analogous to mini-batch sgd?)
    * 2. Repartition to numBags (this is analogous to ensemble averaging?)
    * 3. For each bag we run SGD (observation-wise gradient updates)
    * 4. We then average fitted weights for each feature and return them as updated model
    *
    * @param input     takes a sample fraction and returns collection of examples to be trained in sgd iteration
    * @param sgdParams : SgdParams for training
    * @param modelBC   broadcasted current model (weights)
    * @return
    */
  def sgdTrain(input: Double => RDD[SparseLabeledPoint],
               sgdParams: SgdParams,
               modelBC: Broadcast[AdditiveModel],
               learningRate: Double,
               lossOnly: Boolean = false): AdditiveModel = {
    val model = modelBC.value
    val params = sgdParams.params
    val data = if (lossOnly) {
      input(1.0) // input has already been sampled during split
    } else if (params.partitionSubsample < 1) {
      val numPartitions = input(1.0).partitions.length

      val sampledPartitions = random.shuffle((0 until numPartitions).toList)
        .take((numPartitions * params.partitionSubsample).toInt)
        .toSet

      input(params.subsample)
        .mapPartitionsWithIndex((idx, itr) => if (sampledPartitions.contains(idx)) itr else Iterator.empty)
        .coalesce(params.numBags, params.io.shuffle)
    } else {
      input(params.subsample).coalesce(params.numBags, params.io.shuffle)
    }

    if (!params.io.checkPointDir.isEmpty) {
      data.sparkContext.setCheckpointDir(params.io.checkPointDir)
      data.checkpoint()
    }

    if (lossOnly) {
      data.foreachPartition(sgdPartition(modelBC, sgdParams, learningRate, lossOnly)(0, _))
    } else {
      data.mapPartitionsWithIndex(sgdPartition(modelBC, sgdParams, learningRate, lossOnly))
        .groupByKey()
        // Average the feature functions
        // Average the weights
        .mapValues(aggregateFuncWeights(1.0f / params.numBags, params))
        .collect()
        .foreach {
          case ((familyName, featureName), function) =>
            val family = model.getWeights.get(familyName)
            if (family != null && family.containsKey(featureName)) {
              family.put(featureName, function)
            }
        }
    }

    model
  }

  /**
    * For multiscale feature, we need to resample the model so we can update the model using
    * the particular number of knots
    *
    * @param index     partition index (for multiscale distribution)
    * @param partition list of examples in this partition
    * @param modelBC   broadcasted model weights
    * @param sgdParams : SgdParams for training
    * @return
    */
  def sgdPartition(modelBC: Broadcast[AdditiveModel],
                   sgdParams: SgdParams,
                   learningRate: Double,
                   lossOnly: Boolean)
                  (index: Int,
                   partition: Iterator[SparseLabeledPoint]): Iterator[((String, String), Function)] = {
    val workingModel = modelBC.value.clone()
    val multiscale = sgdParams.params.multiscale

    if (multiscale.nonEmpty && !lossOnly) {
      val newBins = multiscale(index % multiscale.length)

      log.info(s"Resampling to $newBins bins")
      workingModel.getWeightVector.iterator.foreach(_.resample(newBins))
    }

    sgdPartitionInternal(partition, workingModel, sgdParams, learningRate, lossOnly)
  }

  /**
    * Average function weights according to function type. Optionally smooth the weights
    * for spline function.
    *
    * @param functions list of function weights
    * @param scale     scaling factor for aggregation
    * @param params    AdditiveTrainerParams
    * @return
    */
  private def aggregateFuncWeights(scale: Float,
                                   params: AdditiveTrainerParams)
                                  (functions: Iterable[Function]): Function = {
    val head: Function = functions.head
    val output = head.aggregate(functions.asJava, scale, params.numBins)
    output.smooth(params.smoothingTolerance, params.smoothingByPercentage)
    output.LInfinityCap(params.linfinityCap.toFloat)
    output
  }

  /**
    * Actually perform SGD on examples by applying approriate gradient updates according
    * to model specification
    *
    * @param partition    list of examples
    * @param workingModel model to be updated
    * @param sgdParams    SgdParams for training
    * @return
    */
  private def sgdPartitionInternal(partition: Iterator[SparseLabeledPoint],
                                   workingModel: AdditiveModel,
                                   sgdParams: SgdParams,
                                   learningRate: Double,
                                   lossOnly: Boolean): Iterator[((String, String), Function)] = {
    var lossSum: Double = 0.0
    var lossCount: Int = 0

    val params = sgdParams.params
    partition.foreach(example => {
      val lossValue = params.loss.function.update(workingModel, example, params, learningRate, lossOnly)

      lossSum += lossValue
      lossCount += 1
      if (lossCount % params.loss.lossMod == 0) {
        log.info(s"Loss = ${lossSum / params.loss.lossMod}, samples = $lossCount")
        sgdParams.loss += lossSum
        lossSum = 0.0
      }
    })
    sgdParams.loss += lossSum
    sgdParams.exampleCount += lossCount

    if (lossOnly) Iterator.empty
    else
      workingModel.getWeights.entrySet().iterator()
        .flatMap {
          family => family.getValue.entrySet().iterator().map {
            feature => ((family.getKey, feature.getKey), feature.getValue)
          }
        }
  }

  private def modelInitialization(sc: SparkContext, input: Double => RDD[Example],
                                  additiveTrainerParams: AdditiveTrainerParams): AdditiveModel = {
    // sample examples to be used for model initialization
    val params = additiveTrainerParams.init
    log.info(s"Initializing using $params")
    if (params.initModelPath == "" || !PipelineUtil.hdfsFileExists(params.initModelPath)) {
      val newModel = new AdditiveModel()
      initModel(sc, additiveTrainerParams, input, newModel, true)
      setPrior(params.priors, newModel)
      newModel
    } else {
      val newModel = TrainingUtils.loadScoreModel(params.initModelPath)
        .get.asInstanceOf[AdditiveModel]
      if (!params.onlyUseInitModelFunctions) {
        initModel(sc, additiveTrainerParams, input, newModel, false)
      }
      newModel
    }
  }

  // Initializes the model
  private def initModel(sc: SparkContext,
                        additiveTrainerParams: AdditiveTrainerParams,
                        input: Double => RDD[Example],
                        model: AdditiveModel,
                        overwrite: Boolean) = {
    val params = additiveTrainerParams.init
    if (params.nDTreePipelineParams != null) {
      val initExamples = input(params.nDTreePipelineParams.sample)
      val result: Array[((String, String), Either[Array[NDTreeNode], FeatureStats])] = NDTreePipeline.getFeatures(
        sc, initExamples, params.nDTreePipelineParams)
      for (((family, name), feature) <- result) {
        feature match {
          case Left(nodes) =>
            model.addFunction(family, name, new MultiDimensionSpline(nodes), overwrite)
          case Right(stats) =>
            if (stats.spline) {
              val spline = new Spline(stats.min.toFloat, stats.max.toFloat, additiveTrainerParams.numBins)
              model.addFunction(family, name, spline, overwrite)
            } else if (stats.min == stats.max) {
              model.addFunction(family, name, new Point(), overwrite)
            } else {
              model.addFunction(family, name,
                new Linear(stats.min.toFloat, stats.max.toFloat), overwrite)
            }
        }
      }
    } else {
      val initExamples = input(additiveTrainerParams.init.initSubsample)
      initWithoutDynamicBucketModel(additiveTrainerParams, initExamples, model, overwrite)
    }
  }

  // init spline and linear
  private def initWithoutDynamicBucketModel(
                                             additiveTrainerParams: AdditiveTrainerParams,
                                             initExamples: RDD[Example],
                                             model: AdditiveModel,
                                             overwrite: Boolean) = {
    val params = additiveTrainerParams.init
    val linearFeatureFamilies = params.linearFeatureFamilies
    val minMax = TrainingUtils
      .getFeatureStatistics(params.minCount, initExamples, params.initQuantiles)
      .filter(x => x._1._1 != additiveTrainerParams.rankKey)
    log.info("Num features = %d".format(minMax.length))
    val (minMaxLinear, minMaxSpline) = minMax.partition(x => linearFeatureFamilies.contains(x._1._1))
    // add splines
    for (((featureFamily, featureName), stats) <- minMaxSpline) {
      val Seq(min, max) = if (stats.quantiles.isEmpty) Seq(stats.min, stats.max) else stats.quantiles
      if (min == max) {
        model.addFunction(featureFamily, featureName, new Point(), overwrite)
      } else {
        val spline = new Spline(min.toFloat, max.toFloat, additiveTrainerParams.numBins)
        model.addFunction(featureFamily, featureName, spline, overwrite)
      }
    }
    // add linear
    for (((featureFamily, featureName), stats) <- minMaxLinear) {
      // set default linear function as f(x) = 0
      if (stats.min == stats.max) {
        model.addFunction(featureFamily, featureName, new Point(), overwrite)
      } else {
        model.addFunction(featureFamily, featureName,
          new Linear(stats.min.toFloat, stats.max.toFloat), overwrite)
      }
    }
  }

  def deleteSmallFunctions(model: AdditiveModel,
                           linfinityThreshold: Double,
                           remove: Boolean) = {
    val toDelete = scala.collection.mutable.ArrayBuffer[(String, String)]()
    var totalFunction = 0
    model.getWeights.iterator.foreach(family => {
      totalFunction += family._2.size()
      family._2.asScala.foreach(entry => {
        val func: Function = entry._2
        if (func.LInfinityNorm() <= linfinityThreshold) {
          toDelete.append((family._1, entry._1))
        }
      })
    })

    log.info(s"Deleting ${toDelete.size} small functions from $totalFunction")
    toDelete.foreach(entry => {
      val family = model.getWeights.get(entry._1)
      if (family != null && family.containsKey(entry._2)) {
        if (remove) {
          family.remove(entry._2)
        } else {
          family.put(entry._2, new Zero)
        }
      }
    })
  }

  def setPrior(priors: Array[String], model: AdditiveModel): Unit = {
    // set prior for existing functions in the model
    try {
      for (prior <- priors) {
        val tokens: Array[String] = prior.split(",")
        if (tokens.length == 4) {
          val family = tokens(0)
          val name = tokens(1)
          val params = Array(tokens(2).toFloat, tokens(3).toFloat)
          val familyMap = model.getWeights.get(family)
          if (!familyMap.isEmpty) {
            val func: Function = familyMap.get(name)
            if (func != null) {
              log.info("Setting prior %s:%s <- %f to %f".format(family, name, params(0), params(1)))
              func.setPriors(params)
            }
          }
        } else {
          log.error("Incorrect number of parameters for %s".format(prior))
        }
      }
    } catch {
      case _: Throwable => log.info("No prior given")
    }
  }

  def loadTrainingParameters(config: Config, isTraining: Option[Example => Boolean]): AdditiveTrainerParams = {
    val numBins: Int = config.getInt("num_bins")
    val numBags: Int = config.getInt("num_bags")
    val rankKey: String = config.getString("rank_key")
    val dropout: Double = config.getDouble("dropout")
    val subsample: Double = config.getDouble("subsample")
    val validationSubsample: Double = Try(config.getDouble("validation_subsample")).getOrElse(subsample)
    val partitionSubsample: Double = Try(config.getDouble("partition_subsample")).getOrElse(1.0)
    val linfinityCap: Double = config.getDouble("linfinity_cap")
    val smoothingTolerance: Double = config.getDouble("smoothing_tolerance")
    val smoothingByPercentage: Boolean = Try(config.getBoolean("smoothing_by_percentage")).getOrElse(false)
    val linfinityThreshold: Double = config.getDouble("linfinity_threshold")
    val capIterations: Int = Try(config.getInt("cap_iterations")).getOrElse(1)
    val threshold: Double = config.getDouble("rank_threshold")
    val epsilon: Double = Try(config.getDouble("epsilon")).getOrElse(0.0)
    val margin: Double = Try(config.getDouble("margin")).getOrElse(1.0)
    val multiscale: Array[Int] = Try(config.getIntList("multiscale").map(_.intValue()).toArray).getOrElse(Array())

    val classWeights: Map[Int, Float] = Map(-1 -> 1.0f, 1 -> 1.0f) ++
      Try(config.getStringList("class_weights"))
        .map(_.map {
          entry => {
            val Array(clazz, weight) = entry.split(":")
            clazz.toInt -> weight.toFloat
          }
        }.toMap)
        .getOrElse(Map())


    val initModelPath: String = Try(config.getString("init_model")).getOrElse("")
    val onlyUseInitModelFunctions: Boolean = Try(config.getBoolean("only_use_init_model_functions")).getOrElse(false)
    val linearFeatureFamilies: Set[String] = Try(config.getStringList("linear_feature").toSet).getOrElse(Set())
    // if use dynamic_buckets, this is not used
    val priors: Array[String] = Try(config.getStringList("prior").toList.toArray).getOrElse(Array[String]())
    // if use dynamic_buckets, this is not used
    val minCount: Int = Try(config.getInt("min_count")).getOrElse(0)
    val nDTreePipelineParams = Try(config.getConfig("dynamic_buckets"))
      .map(NDTreePipeline.getNDTreePipelineParams)
      .getOrElse(null)
    val initQuantiles: Seq[Double] = Try(
      config.getDoubleList("init_quantiles").map(_.doubleValue())
    ).getOrElse(Nil)
    assert(initQuantiles.isEmpty || initQuantiles.length == 2, "initialization quantiles must be length of 2.")
    val initSubsample = Try(config.getDouble("init_subsample")).getOrElse(subsample)
    val initParams = InitParams(initModelPath, onlyUseInitModelFunctions, linearFeatureFamilies, priors, minCount, nDTreePipelineParams, initQuantiles, initSubsample)

    val loss: LossFunction = LossFunctions.withName(config.getString("loss").toUpperCase)
    val lossMod: Int = Try(config.getInt("loss_mod")).getOrElse(100)
    val earlyStopping = Try(config.getInt("early_stopping")).getOrElse(0)
    val convergenceTolerance = Try(config.getDouble("convergence_tolerance")).getOrElse(0.0)
    val lossParams = LossParams(loss, lossMod, earlyStopping, convergenceTolerance, isTraining)

    val learningRate: Double = config.getDouble("learning_rate")
    val minLearningRate: Double = Try(config.getDouble("min_learning_rate")).getOrElse(0)
    val learningRateDecay: Double = Try(config.getDouble("learning_rate_decay")).getOrElse(1)
    val decayByValidationLoss = Try(config.getBoolean("decay_by_validation_loss")).getOrElse(true)
    val decayLookbackIterations = Try(config.getInt("decay_lookback_iterations")).getOrElse(3)
    val learningRateParams = LearningRateParams(learningRate, learningRateDecay, minLearningRate, decayByValidationLoss, decayLookbackIterations)

    val modelOutput = Try(config.getString("model_output")).getOrElse("")
    val shuffle: Boolean = Try(config.getBoolean("shuffle")).getOrElse(true)
    val storageLevel = StorageLevel.fromString(Try(config.getString("storage_level")).getOrElse("NONE").toUpperCase())
    val checkPointDir = Try(config.getString("check_point_dir")).getOrElse("")
    val saveModelEachIteration = Try(config.getBoolean("save_model_each_iteration")).getOrElse(false)
    val ioParams = IOParams(modelOutput, shuffle, checkPointDir, storageLevel, saveModelEachIteration)

    AdditiveTrainerParams(
      numBins,
      numBags,
      rankKey,
      lossParams,
      learningRateParams,
      dropout,
      subsample,
      validationSubsample,
      partitionSubsample,
      margin,
      multiscale,
      smoothingTolerance,
      smoothingByPercentage,
      linfinityThreshold,
      linfinityCap,
      capIterations,
      threshold,
      epsilon,
      initParams,
      classWeights,
      ioParams)
  }

  def trainAndSaveToFile(sc: SparkContext, input: RDD[Example], config: Config, key: String) = {
    trainAndSaveToFileEarlySample(sc, (frac: Double) => input.sample(false, frac), config, key)
  }

  /**
    * Entry point to train and persist model on disk
    *
    * This version allows sample to be pushed down in the Example loading process.
    * One use case is to avoid deserialization when examples are discarded by sampling.
    *
    * @note Care should be taken when caching dataset as the order of cache and sample call will determine the proportion
    *       of dataset be cached and whether each reference will result in a new set of sample.
    * @param sampleInput a function takes sampling fraction and returns sampled dataset
    */
  def trainAndSaveToFileEarlySample(sc: SparkContext,
                                    sampleInput: Double => RDD[Example],
                                    config: Config,
                                    key: String,
                                    isTraining: Option[Example => Boolean] = None) = {
    val model = train(sc, sampleInput, config, key, isTraining)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }
}
