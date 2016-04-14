package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core._
import com.airbnb.aerosolve.core.features.{Family, Feature, FeatureRegistry, MultiFamilyVector}
import com.airbnb.aerosolve.core.functions.{Function, Linear, Spline}
import com.airbnb.aerosolve.core.models.AdditiveModel
import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
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

  case class AdditiveTrainerParams(numBins: Int,
                                   numBags: Int,
                                   labelFamily: Family,
                                   loss: String,
                                   minCount: Int,
                                   learningRate: Double,
                                   dropout: Double,
                                   subsample: Double,
                                   margin: Double,
                                   multiscale: Array[Int],
                                   smoothingTolerance: Double,
                                   linfinityThreshold: Double,
                                   linfinityCap: Double,
                                   threshold: Double,
                                   lossMod: Int,
                                   isRanking: Boolean, // If we have a list based ranking loss
                                   rankMargin: Double, // The margin for ranking loss
                                   epsilon: Double, // epsilon used in epsilon-insensitive loss for regression training
                                   initModelPath: String,
                                   linearFeatureFamilies: Array[Family],
                                   priors: Array[String],
                                   registry: FeatureRegistry)

  def train(sc: SparkContext,
            input: RDD[Example],
            config: Config,
            key: String,
            registry: FeatureRegistry): AdditiveModel = {
    val trainConfig = config.getConfig(key)
    val iterations: Int = trainConfig.getInt("iterations")
    val params = loadTrainingParameters(trainConfig, registry)
    val transformed = transformExamples(input, config, key, params)
    val output = config.getString(key + ".model_output")
    log.info("Training using " + params.loss)

    val paramsBC = sc.broadcast(params)
    var model = modelInitialization(transformed, params)
    for (i <- 1 to iterations) {
      log.info(s"Iteration $i")
      val modelBC = sc.broadcast(model)
      model = sgdTrain(transformed, paramsBC, modelBC)
      modelBC.unpersist()

      TrainingUtils.saveModel(model, output)
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
    * @param input    collection of examples to be trained in sgd iteration
    * @param paramsBC broadcasted model params
    * @param modelBC  broadcasted current model (weights)
    * @return
    */
  def sgdTrain(input: RDD[Example],
               paramsBC: Broadcast[AdditiveTrainerParams],
               modelBC: Broadcast[AdditiveModel]): AdditiveModel = {
    val model = modelBC.value
    val params = paramsBC.value

    // This returns the entire additive model in order to support familyWeights.
    val fittedModels: RDD[AdditiveModel] = input
      .sample(false, params.subsample)
      .coalesce(params.numBags, true)
      .mapPartitionsWithIndex((index, partition) => sgdPartition(index, partition, modelBC, paramsBC))

    val weightsPerFeature: Array[(AnyRef, Function)] = fittedModels
      .flatMap(model => model.weights.asScala.iterator ++ model.familyWeights.asScala.iterator)
      // TODO (Brad): This should be a reduceByKey. It's complicated by the aggregate method in
      // Function. We'd need to aggregate one function at a time. But it should be possible and
      // would likely help a lot with performance.
      .groupByKey()
      // Average the feature functions
      // Average the weights
      .mapValues(x => {
        val scale = 1.0d / paramsBC.value.numBags
        aggregateFuncWeights(x, scale, paramsBC.value.numBins, paramsBC.value.smoothingTolerance)
      })
      .collect()

    weightsPerFeature
      .foreach {
        case (key: Family, function: Function) => model.familyWeights.replace(key, function)
        case (key: Feature, function: Function) => model.weights.replace(key, function)
      }


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
    * @param paramsBC  broadcasted model params
    * @return
    */
  def sgdPartition(index: Int,
                   partition: Iterator[Example],
                   modelBC: Broadcast[AdditiveModel],
                   paramsBC: Broadcast[AdditiveTrainerParams]): Iterator[AdditiveModel] = {
    val workingModel = modelBC.value
    val params = paramsBC.value
    val multiscale = params.multiscale

    if (multiscale.nonEmpty) {
      val newBins = multiscale(index % multiscale.length)

      log.info(s"Resampling to $newBins bins")
      workingModel.weights.values.foreach(_.resample(newBins))
    }

    val output = sgdPartitionInternal(partition, workingModel, params)
    Set(output).iterator
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
                                   scale: Double,
                                   numBins: Int,
                                   smoothingTolerance: Double): Function = {
    val head: Function = input.head
    // TODO (Forest): revisit asJava performance impact
    val output = head.aggregate(input.asJava, scale, numBins)
    output.smooth(smoothingTolerance)
    output
  }

  /**
    * Actually perform SGD on examples by applying appropriate gradient updates according
    * to model specification
    *
    * @param partition    list of examples
    * @param workingModel model to be updated
    * @param params       model parameters
    * @return
    */
  private def sgdPartitionInternal(partition: Iterator[Example],
                                   workingModel: AdditiveModel,
                                   params: AdditiveTrainerParams): AdditiveModel = {
    var lossSum: Double = 0.0
    var lossCount: Int = 0
    partition.foreach(example => {
      lossSum += pointwiseLoss(example.only(), workingModel, params.loss, params)
      lossCount = lossCount + 1
      if (lossCount % params.lossMod == 0) {
        log.info(s"Loss = ${lossSum / params.lossMod.toDouble}, samples = $lossCount")
        lossSum = 0.0
      }
    })
    workingModel
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
  def pointwiseLoss(fv: MultiFamilyVector,
                    workingModel: AdditiveModel,
                    loss: String,
                    params: AdditiveTrainerParams): Double = {
    val label: Double = if (loss == "regression") {
      TrainingUtils.getLabel(fv, params.labelFamily)
    } else {
      TrainingUtils.getLabel(fv, params.labelFamily, params.threshold)
    }

    val lossFunction = loss match {
      case "logistic" => logisticLoss _
      case "hinge" => hingeLoss _
      case "regression" => regressionLoss _
    }
    updateWithLossFunction(workingModel, fv, label, lossFunction, params)
  }

  // http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
  // We rescale by 1 / p so that at inference time we don't have to scale by p.
  // In our case p = 1.0 - dropout rate
  def updateWithLossFunction(model: AdditiveModel,
         fv: MultiFamilyVector,
         label: Double,
         lossAndGradientFunc: (Double, Double, AdditiveTrainerParams) => (Double, Double),
         params: AdditiveTrainerParams): Double = {
    val newVec = fv.withFamilyDropout(params.dropout)
    val prediction = model.scoreItem(newVec) / (1.0 - params.dropout)
    val (loss, grad) = lossAndGradientFunc(prediction, label, params)
    val gradWithLearningRate = grad * params.learningRate
    if (gradWithLearningRate != 0.0) {
      model.update(gradWithLearningRate,
                   params.linfinityCap,
                   fv)
    }
    loss
  }

  // TODO (Brad): I refactored out the loss functions to reduce code duplication but I'm a bit
  // concerned I made a logical error.  Would appreciate a close look.
  def logisticLoss(prediction: Double, label: Double, params: AdditiveTrainerParams): (Double, Double) = {
    val corr = scala.math.min(10.0, label * prediction)
    val expCorr = scala.math.exp(corr)
    val loss = scala.math.log(1.0 + 1.0 / expCorr)
    val grad = -label / (1.0 + expCorr)
    (loss, grad)
  }

  def hingeLoss(prediction: Double, label: Double, params : AdditiveTrainerParams): (Double, Double) = {
    val loss = scala.math.max(0.0, params.margin - label * prediction)
    val grad = if (loss > 0.0) -label else 0.0
    (loss, grad)
  }

  def regressionLoss(prediction: Double, label: Double, params : AdditiveTrainerParams): (Double, Double) = {
    val loss = math.abs(prediction - label)
    val grad = if (prediction - label > params.epsilon) 1.0
      else if (prediction - label < -params.epsilon) -1.0
      else 0.0
    (loss, grad)
  }

  private def transformExamples(input: RDD[Example],
                                config: Config,
                                key: String,
                                params: AdditiveTrainerParams): RDD[Example] = {
    if (params.isRanking) {
      LinearRankerUtils.transformExamples(input, config, key, params.registry)
    } else {
      LinearRankerUtils.makePointwiseFloat(input, config, key, params.registry)
    }
  }

  private def modelInitialization(input: RDD[Example],
                                  params: AdditiveTrainerParams): AdditiveModel = {
    // add functions to additive model
    val initialModel = if (params.initModelPath == "") {
      None
    } else {
      TrainingUtils.loadScoreModel(params.initModelPath, params.registry)
    }

    // sample examples to be used for model initialization
    val initExamples = input.sample(false, params.subsample)
    if (initialModel.isDefined) {
      val newModel = initialModel.get.asInstanceOf[AdditiveModel]
      initModel(params.minCount, params, initExamples, newModel, false)
      newModel
    } else {
      val newModel = new AdditiveModel(params.registry)
      initModel(params.minCount, params, initExamples, newModel, true)
      setPrior(params.priors, newModel)
      newModel
    }
  }

  // Initializes the model
  private def initModel(minCount: Int,
                        params: AdditiveTrainerParams,
                        examples: RDD[Example],
                        model: AdditiveModel,
                        overwrite: Boolean) = {
    val linearFeatureFamilies = params.linearFeatureFamilies
    val minMax = TrainingUtils
      .getFeatureStatistics(minCount, examples)
      .filter{ case (feature:Feature, _) => feature.family != params.labelFamily }

    log.info("Num features = %d".format(minMax.length))

    minMax.foreach{ case (feature:Feature, stats) =>
      val function = if (linearFeatureFamilies.contains(feature.family)) {
        // set default linear function as f(x) = 0
        new Linear(stats.min, stats.max)
      } else {
        new Spline(stats.min, stats.max, params.numBins)
      }
      model.addFunction(feature, function, overwrite)
    }
  }

  def deleteSmallFunctions(model: AdditiveModel,
                           linfinityThreshold: Double) = {
    val newWeights = model.weights.asScala.filter {
      case (_, func) => func.LInfinityNorm() >= linfinityThreshold
    }.asJava
    model.weights(newWeights)
  }

  def setPrior(priors: Array[String], model: AdditiveModel): Unit = {
    // set prior for existing functions in the model
    try {
      for (prior <- priors) {
        val tokens: Array[String] = prior.split(",")
        if (tokens.length == 4) {
          val family = tokens(0)
          val name = tokens(1)
          val params = Array(tokens(2).toDouble, tokens(3).toDouble)
          val feature = model.registry.feature(family, name)
          val func: Function = model.weights.get(feature)
          if (func != null) {
            log.info("Setting prior %s:%s <- %f to %f".format(family, name, params(0), params(1)))
            func.setPriors(params)
          }
        } else {
          log.error("Incorrect number of parameters for %s".format(prior))
        }
      }
    } catch {
      case _: Throwable => log.info("No prior given")
    }
  }

  def loadTrainingParameters(config: Config,
                             registry: FeatureRegistry): AdditiveTrainerParams = {
    val loss: String = config.getString("loss")
    val isRanking = loss match {
      case "logistic" => false
      case "hinge" => false
      case "regression" => false
      case _ =>
        throw new IllegalArgumentException("Unknown loss function %s".format(loss))
    }
    val numBins: Int = config.getInt("num_bins")
    val numBags: Int = config.getInt("num_bags")
    val labelFamily: Family = registry.family(config.getString("rank_key"))
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
    val minCount: Int = config.getInt("min_count")
    val linearFeaturePath = "linear_feature"
    val linearFeatureFamilies: Array[Family] = if (config.hasPath(linearFeaturePath)) { 
        config.getStringList(linearFeaturePath)
          .map(familyName => registry.family(familyName)).toList.toArray
      } else Array[Family]()
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

    val rankMargin: Double = Try(config.getDouble("rank_margin")).getOrElse(0.5)

    AdditiveTrainerParams(
      numBins,
      numBags,
      labelFamily,
      loss,
      minCount,
      learningRate,
      dropout,
      subsample,
      margin,
      multiscale,
      smoothingTolerance,
      linfinityThreshold,
      linfinityCap,
      threshold,
      lossMod,
      isRanking,
      rankMargin,
      epsilon,
      initModelPath,
      linearFeatureFamilies,
      priors,
      registry)
  }

  def trainAndSaveToFile(sc: SparkContext,
                         input: RDD[Example],
                         config: Config,
                         key: String,
                          registry: FeatureRegistry) = {
    val model = train(sc, input, config, key, registry)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }
}
