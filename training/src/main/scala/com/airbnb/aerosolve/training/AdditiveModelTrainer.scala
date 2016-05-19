package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core._
import com.airbnb.aerosolve.core.function._
import com.airbnb.aerosolve.core.models.{AdditiveModel, NDTreeModel}
import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.training.pipeline.NDTreePipeline
import com.airbnb.aerosolve.training.pipeline.NDTreePipeline.{FeatureStats, NDTreePipelineParams}
import com.typesafe.config.Config
import org.apache.spark.{Accumulator, SparkContext}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.util.Try

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

  case class SgdParams(paramsBC: Broadcast[AdditiveTrainerParams],
                      exampleCount: Accumulator[Long],
                      loss: Accumulator[Double])

  case class LossParams(function: String,
                        lossMod: Int,
                        useBestLoss: Boolean,
                        minLoss: Double)

  case class AdditiveTrainerParams(numBins: Int,
                                   numBags: Int,
                                   rankKey: String,
                                   minCount: Int,
                                   loss: LossParams,
                                   learningRate: Double,
                                   dropout: Double,
                                   subsample: Double,
                                   margin: Double,
                                   multiscale: Array[Int],
                                   smoothingTolerance: Double,
                                   linfinityThreshold: Double,
                                   linfinityCap: Double,
                                   threshold: Double,
                                   epsilon: Double, // epsilon used in epsilon-insensitive loss for regression training
                                   initModelPath: String,
                                   linearFeatureFamilies: java.util.List[String],
                                   priors: Array[String],
                                   nDTreePipelineParams: NDTreePipelineParams,
                                   classWeights: Map[Int, Float])

  def train(sc: SparkContext, input: RDD[Example], config: Config, key: String): AdditiveModel =
    train(sc, (frac: Double) => input.sample(false, frac), config, key)

  def train(sc: SparkContext,
            input: Double => RDD[Example],
            config: Config,
            key: String): AdditiveModel = {
    val trainConfig = config.getConfig(key)
    val iterations: Int = trainConfig.getInt("iterations")
    val params = loadTrainingParameters(trainConfig)

    // sample before we transform as it can be very expensive especially for crossing
    // NB: this assumes we don't add/remove observations during transformation
    val transformed = (frac: Double) => LinearRankerUtils.makePointwiseFloat(input(frac), config, key)
    val output = config.getString(key + ".model_output")
    log.info("Training using " + params.loss)

    val paramsBC = sc.broadcast(params)
    var model = modelInitialization(sc, transformed, params)
    var loss = Double.MaxValue
    for (i <- 1 to iterations
         if loss >= params.loss.minLoss) {
      log.info(s"Iteration $i")
      val sgdParams = SgdParams(
        paramsBC,
        sc.accumulator(0),
        sc.accumulator(0))
      val modelBC = sc.broadcast(model)
      model = sgdTrain(transformed, sgdParams, modelBC)
      modelBC.unpersist()
      val newLoss = sgdParams.loss.value/sgdParams.exampleCount.value
      if (params.loss.useBestLoss && newLoss < loss) {
        TrainingUtils.saveModel(model, output)
        log.info(s"iterations $i useBestLoss ThisRoundLoss = $newLoss count = $sgdParams.exampleCount.value")
        loss = newLoss
      } else {
        log.info(s"iterations $i ThisRoundLoss = $newLoss count = $sgdParams.exampleCount.value")
      }
    }
    model
  }

  /**
    * During each iteration, we:
    *
    * 1. Sample dataset with subsample (this is analogous to mini-batch sgd?)
    * 2. Repartition to numBags (this is analogous to ensemble averaging?)
    * 3. For each bag we run SGD (observation-wise gradient updates)
    * 4. We then average fitted weights for each feature and return them as updated model
    *
    * @param input    takes a sample fraction and returns collection of examples to be trained in sgd iteration
    * @param sgdParams: SgdParams for training
    * @param modelBC  broadcasted current model (weights)
    * @return
    */
  def sgdTrain(input: Double => RDD[Example],
               sgdParams: SgdParams,
               modelBC: Broadcast[AdditiveModel]): AdditiveModel = {
    val model = modelBC.value
    val params = sgdParams.paramsBC.value

    input(params.subsample)
      .coalesce(params.numBags, true)
      .mapPartitionsWithIndex((index, partition) => sgdPartition(index, partition, modelBC, sgdParams))
      .groupByKey()
      // Average the feature functions
      // Average the weights
      .mapValues(x => {
      val scale = 1.0f / params.numBags.toFloat
      aggregateFuncWeights(x, scale, params.numBins, params.smoothingTolerance.toFloat)
    })
      .collect()
      .foreach(entry => {
        val family = model.getWeights.get(entry._1._1)
        if (family != null && family.containsKey(entry._1._2)) {
          family.put(entry._1._2, entry._2)
        }
      })

    deleteSmallFunctions(model, params.linfinityThreshold)
    model
  }

  /**
    * For multiscale feature, we need to resample the model so we can update the model using
    * the particular number of knots
    *
    * @param index     partition index (for multiscale distribution)
    * @param partition list of examples in this partition
    * @param modelBC   broadcasted model weights
    * @param sgdParams: SgdParams for training
    * @return
    */
  def sgdPartition(index: Int,
                   partition: Iterator[Example],
                   modelBC: Broadcast[AdditiveModel],
                   sgdParams: SgdParams): Iterator[((String, String), Function)] = {
    val workingModel = modelBC.value.clone()
    val multiscale = sgdParams.paramsBC.value.multiscale

    if (multiscale.nonEmpty) {
      val newBins = multiscale(index % multiscale.length)

      log.info(s"Resampling to $newBins bins")
      for(family <- workingModel.getWeights.values) {
        for(feature <- family.values) {
          feature.resample(newBins)
        }
      }
    }

    val output = sgdPartitionInternal(partition, workingModel, sgdParams)
    output.iterator
  }

  /**
    * Average function weights according to function type. Optionally smooth the weights
    * for spline function.
    *
    * @param input              list of function weights
    * @param scale              scaling factor for aggregation
    * @param numBins            number of bins for final weights
    * @param smoothingTolerance smoothing tolerance for spline
    * @return
    */
  private def aggregateFuncWeights(input: Iterable[Function],
                                   scale: Float,
                                   numBins: Int,
                                   smoothingTolerance: Float): Function = {
    val head: Function = input.head
    // TODO: revisit asJava performance impact
    val output = head.aggregate(input.asJava, scale, numBins)
    output.smooth(smoothingTolerance)
    output
  }

  /**
    * Actually perform SGD on examples by applying approriate gradient updates according
    * to model specification
    *
    * @param partition    list of examples
    * @param workingModel model to be updated
    * @param sgdParams: SgdParams for training
    * @return
    */
  private def sgdPartitionInternal(partition: Iterator[Example],
                                   workingModel: AdditiveModel,
                                   sgdParams: SgdParams): mutable.HashMap[(String, String), Function] = {
    var lossTotal: Double = 0.0
    var lossSum: Double = 0.0
    var lossCount: Int = 0
    val params = sgdParams.paramsBC.value
    partition.foreach(example => {
      val lossValue = pointwiseLoss(example.example.get(0), workingModel, params.loss.function, params)
      lossSum += lossValue
      lossTotal += lossValue
      lossCount = lossCount + 1
      if (lossCount % params.loss.lossMod == 0) {
        log.info(s"Loss = ${lossSum / params.loss.lossMod.toDouble}, samples = $lossCount")
        lossSum = 0.0
      }
    })
    val output = mutable.HashMap[(String, String), Function]()
    // TODO: weights should be a vector instead of stored in hashmap
    workingModel
      .getWeights
      .foreach(family => {
        family._2.foreach(feature => {
          output.put((family._1, feature._1), feature._2)
        })
      })
    sgdParams.exampleCount.+=(lossCount)
    sgdParams.loss.+=(lossTotal)
    output
  }

  /**
    * Compute loss for a single observation and update model weights during the process
    *
    * @param fv           observation
    * @param workingModel model to be updated
    * @param loss         loss type
    * @param params       model params
    * @return
    */
  def pointwiseLoss(fv: FeatureVector,
                    workingModel: AdditiveModel,
                    loss: String,
                    params: AdditiveTrainerParams): Double = {
    val label: Double = if (loss == "regression") {
      TrainingUtils.getLabel(fv, params.rankKey)
    } else {
      TrainingUtils.getLabel(fv, params.rankKey, params.threshold)
    }

    loss match {
      case "logistic" => updateLogistic(workingModel, fv, label, params)
      case "hinge" => updateHinge(workingModel, fv, label, params)
      case "regression" => updateRegressor(workingModel, fv, label, params)
    }
  }

  // http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
  // We rescale by 1 / p so that at inference time we don't have to scale by p.
  // In our case p = 1.0 - dropout rate
  def updateLogistic(model: AdditiveModel,
                     fv: FeatureVector,
                     label: Double,
                     params: AdditiveTrainerParams): Double = {
    val classWeights = params.classWeights
    val flatFeatures = Util.flattenFeatureWithDropout(fv, params.dropout)
    // only MultiDimensionSpline use denseFeatures for now
    val denseFeatures = MultiDimensionSpline.featureDropout(fv, params.dropout)
    val prediction = (model.scoreFlatFeatures(flatFeatures) +
      model.scoreDenseFeatures(denseFeatures)) /
      (1.0 - params.dropout)
    // To prevent blowup.
    val corr = scala.math.min(10.0, label * prediction)
    val expCorr = scala.math.exp(corr)
    val loss = scala.math.log(1.0 + 1.0 / expCorr)
    val grad = -label / (1.0 + expCorr)
    val gradWithLearningRate = grad.toFloat * params.learningRate.toFloat * classWeights(label.toInt)
    model.update(gradWithLearningRate,
      params.linfinityCap.toFloat,
      flatFeatures)
    model.updateDense(gradWithLearningRate,
      params.linfinityCap.toFloat,
      denseFeatures)
    loss
  }

  def updateHinge(model: AdditiveModel,
                  fv: FeatureVector,
                  label: Double,
                  params: AdditiveTrainerParams): Double = {
    val classWeights = params.classWeights
    val flatFeatures = Util.flattenFeatureWithDropout(fv, params.dropout)
    // only MultiDimensionSpline use denseFeatures for now
    val denseFeatures = MultiDimensionSpline.featureDropout(fv, params.dropout)
    val prediction = (model.scoreFlatFeatures(flatFeatures) +
      model.scoreDenseFeatures(denseFeatures)) /
      (1.0 - params.dropout)
    val loss = scala.math.max(0.0, params.margin - label * prediction)
    if (loss > 0.0) {
      val gradWithLearningRate = -label.toFloat * params.learningRate.toFloat * classWeights(label.toInt)
      model.update(gradWithLearningRate,
        params.linfinityCap.toFloat,
        flatFeatures)
      model.updateDense(gradWithLearningRate,
        params.linfinityCap.toFloat,
        denseFeatures)
    }
    loss
  }

  def updateRegressor(model: AdditiveModel,
                      fv: FeatureVector,
                      label: Double,
                      params: AdditiveTrainerParams): Double = {
    val flatFeatures = Util.flattenFeatureWithDropout(fv, params.dropout)
    // only MultiDimensionSpline use denseFeatures for now
    val denseFeatures = MultiDimensionSpline.featureDropout(fv, params.dropout)
    val prediction = (model.scoreFlatFeatures(flatFeatures) +
      model.scoreDenseFeatures(denseFeatures)) /
      (1.0 - params.dropout)
    // absolute difference
    val loss = math.abs(prediction - label)
    if (prediction - label > params.epsilon) {
      model.update(params.learningRate.toFloat,
        params.linfinityCap.toFloat, flatFeatures)
      model.updateDense(params.learningRate.toFloat,
        params.linfinityCap.toFloat, denseFeatures)
    } else if (prediction - label < -params.epsilon) {
      model.update(-params.learningRate.toFloat,
        params.linfinityCap.toFloat, flatFeatures)
      model.updateDense(-params.learningRate.toFloat,
        params.linfinityCap.toFloat, denseFeatures)
    }
    loss
  }

  private def modelInitialization(sc: SparkContext, input: Double => RDD[Example],
                                  params: AdditiveTrainerParams): AdditiveModel = {
    // sample examples to be used for model initialization
    if (params.initModelPath == "") {
      val newModel = new AdditiveModel()
      initModel(sc, params, input, newModel, true)
      setPrior(params.priors, newModel)
      newModel
    } else {
      val newModel = TrainingUtils.loadScoreModel(params.initModelPath)
        .get.asInstanceOf[AdditiveModel]
      initModel(sc, params, input, newModel, false)
      newModel
    }
  }

  // Initializes the model
  private def initModel(sc: SparkContext,
                        params: AdditiveTrainerParams,
                        input: Double => RDD[Example],
                        model: AdditiveModel,
                        overwrite: Boolean) = {
    if (params.nDTreePipelineParams != null) {
      val initExamples = input(params.nDTreePipelineParams.sample)
      val linearFeatureFamilies = params.linearFeatureFamilies
      val result: Array[((String, String), Either[NDTreeModel, FeatureStats])] = NDTreePipeline.getFeatures(
        sc, initExamples, params.nDTreePipelineParams)
      for (((family, name), feature) <- result) {
        feature match {
          case Left(ndTreeModel) => {
            model.addFunction(family, name, new  MultiDimensionSpline(ndTreeModel), overwrite)
          }
          case Right(stats) => {
            if (stats.min == stats.max) {
              model.addFunction(family, name, new Point(stats.min.toFloat), overwrite)
            } else {
              model.addFunction(family, name,
                new Linear(stats.min.toFloat, stats.max.toFloat), overwrite)
            }
          }
        }
      }
    } else {
      val initExamples = input(params.subsample)
      initWithoutDynamicBucketModel(params, initExamples, model, overwrite)
    }
  }

  // init spline and linear
  private def initWithoutDynamicBucketModel(params: AdditiveTrainerParams,
                        initExamples: RDD[Example],
                        model: AdditiveModel,
                        overwrite: Boolean) = {
    val linearFeatureFamilies = params.linearFeatureFamilies
    val minMax = TrainingUtils
      .getFeatureStatistics(params.minCount, initExamples)
      .filter(x => x._1._1 != params.rankKey)
    log.info("Num features = %d".format(minMax.length))
    val minMaxSpline = minMax.filter(x => !linearFeatureFamilies.contains(x._1._1))
    val minMaxLinear = minMax.filter(x => linearFeatureFamilies.contains(x._1._1))
    // add splines
    for (((featureFamily, featureName), stats) <- minMaxSpline) {
      val spline = new Spline(stats.min.toFloat, stats.max.toFloat, params.numBins)
      model.addFunction(featureFamily, featureName, spline, overwrite)
    }
    // add linear
    for (((featureFamily, featureName), stats) <- minMaxLinear) {
      // set default linear function as f(x) = 0
      if (stats.min == stats.max) {
        model.addFunction(featureFamily, featureName, new Point(stats.min.toFloat), overwrite)
      } else {
        model.addFunction(featureFamily, featureName,
          new Linear(stats.min.toFloat, stats.max.toFloat), overwrite)
      }
    }
  }

  def deleteSmallFunctions(model: AdditiveModel,
                           linfinityThreshold: Double) = {
    val toDelete = scala.collection.mutable.ArrayBuffer[(String, String)]()

    model.getWeights.asScala.foreach(family => {
      family._2.asScala.foreach(entry => {
        val func: Function = entry._2
        if (func.LInfinityNorm() < linfinityThreshold) {
          toDelete.append((family._1, entry._1))
        }
      })
    })

    log.info("Deleting %d small functions".format(toDelete.size))
    toDelete.foreach(entry => {
      val family = model.getWeights.get(entry._1)
      if (family != null && family.containsKey(entry._2)) {
        family.remove(entry._2)
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

  def loadTrainingParameters(config: Config): AdditiveTrainerParams = {
    val loss: String = config.getString("loss")
    val numBins: Int = config.getInt("num_bins")
    val numBags: Int = config.getInt("num_bags")
    val rankKey: String = config.getString("rank_key")
    val learningRate: Double = config.getDouble("learning_rate")
    val dropout: Double = config.getDouble("dropout")
    val subsample: Double = config.getDouble("subsample")
    val linfinityCap: Double = config.getDouble("linfinity_cap")
    val smoothingTolerance: Double = config.getDouble("smoothing_tolerance")
    val linfinityThreshold: Double = config.getDouble("linfinity_threshold")
    val initModelPath: String = Try {
      config.getString("init_model")
    }.getOrElse("")
    val threshold: Double = config.getDouble("rank_threshold")
    val epsilon: Double = Try {
      config.getDouble("epsilon")
    }.getOrElse(0.0)
    // if use dynamic_buckets, this is not used
    val minCount: Int = Try {config.getInt("min_count")}.getOrElse(0)
    // if use dynamic_buckets, this is not used
    val linearFeatureFamilies: java.util.List[String] = Try(
      config.getStringList("linear_feature"))
      .getOrElse[java.util.List[String]](List.empty.asJava)
    val lossMod: Int = Try {
      config.getInt("loss_mod")
    }.getOrElse(100)
    val priors: Array[String] = Try(
      config.getStringList("prior").toList.toArray)
      .getOrElse(Array[String]())

    val margin: Double = Try(config.getDouble("margin")).getOrElse(1.0)

    val multiscale: Array[Int] = Try(
      config.getIntList("multiscale").asScala.map(x => x.toInt).toArray)
      .getOrElse(Array[Int]())
    val dynamicBucketsConfig = Try(Some(config.getConfig("dynamic_buckets"))).getOrElse(None)
    val options = if (dynamicBucketsConfig.nonEmpty) {
      val cfg = dynamicBucketsConfig.get
      NDTreePipeline.getNDTreePipelineParams(cfg)
    } else {
      null
    }

    val minLoss: Double = Try(config.getDouble("min_loss")).getOrElse(0)
    val useBestLoss: Boolean = Try(config.getBoolean("use_best_loss")).getOrElse(false)

    val classWeights: mutable.Map[Int, Float] = mutable.Map(-1 -> 1.0f, 1 -> 1.0f)
    Try(config.getStringList("class_weights").toList.toArray)
      .getOrElse(Array[String]())
      .foreach( weight => {
        val str = weight.split(":")
        val cls = str(0).toInt
        val w = str(1).toFloat
        classWeights += (cls -> w)
      })

    val lossParams = LossParams(loss, lossMod, useBestLoss, minLoss)

    AdditiveTrainerParams(
      numBins,
      numBags,
      rankKey,
      minCount,
      lossParams,
      learningRate,
      dropout,
      subsample,
      margin,
      multiscale,
      smoothingTolerance,
      linfinityThreshold,
      linfinityCap,
      threshold,
      epsilon,
      initModelPath,
      linearFeatureFamilies,
      priors,
      options,
      classWeights.toMap)
  }

  def trainAndSaveToFile(sc: SparkContext,
                         input: RDD[Example],
                         config: Config,
                         key: String) = {
    trainAndSaveToFileEarlySample(sc, (frac: Double) => input.sample(false, frac), config, key)
  }

  /**
    * Entry point to train and persist model on disk
    *
    * This version allows sample to be pushed down in the Example loading process.
    * One use case is to avoid deserialization when examples are discarded by sampling.
    *
    * @note Care should be taken when caching dataset as the order of cache and sample call will determine the proportion
    * of dataset be cached and whether each reference will result in a new set of sample.
    * @param sampleInput a function takes sampling fraction and returns sampled dataset
    */
  def trainAndSaveToFileEarlySample(sc: SparkContext,
                         sampleInput: Double => RDD[Example],
                         config: Config,
                         key: String) = {
    val model = train(sc, sampleInput, config, key)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }
}
