package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core._
import com.airbnb.aerosolve.core.function.{Function, Linear, Spline}
import com.airbnb.aerosolve.core.models.AdditiveModel
import com.airbnb.aerosolve.core.util.Util
import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.mutable.HashMap
import scala.util.Try

/**
 * Additive Model Trainer
 * By default, we use a spline function to represent a float feature; use linear function to represent a string feature.
 * Additionally, float features that are specified as 'linear_feature' in config are also represented by a linear function.
 */

object AdditiveModelTrainer {
  private final val log: Logger = LoggerFactory.getLogger("AdditiveModelTrainer")
  case class AdditiveTrainerParams(numBins : Int,
                                   numBags : Int,
                                   rankKey : String,
                                   loss : String,
                                   minCount : Int,
                                   learningRate : Double,
                                   dropout : Double,
                                   subsample : Double,
                                   margin : Double,
                                   multiscale : Array[Int],
                                   smoothingTolerance : Double,
                                   linfinityThreshold : Double,
                                   linfinityCap : Double,
                                   threshold : Double,
                                   lossMod : Int,
                                   isRanking : Boolean,    // If we have a list based ranking loss
                                   rankMargin : Double,    // The margin for ranking loss
                                   epsilon : Double,       // epsilon used in epsilon-insensitive loss for regression training
                                   initModelPath: String,
                                   linearFeatureFamilies: Array[String],
                                   priors: Array[String])

  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String) : AdditiveModel = {
    val trainConfig = config.getConfig(key)
    val iterations : Int = trainConfig.getInt("iterations")
    val params = loadTrainingParameters(trainConfig)
    val transformed = transformExamples(input, config, key, params)
    var model = modelInitialization(transformed, params)
    val output = config.getString(key + ".model_output")
    log.info("Training using " + params.loss)
    for (i <- 1 to iterations) {
      log.info("Iteration %d".format(i))
      model = if (params.multiscale.length == 0) {
        sgdTrain(
          sc,
          transformed,
          params,
          model,
          output)
      } else {
        sgdMultiscaleTrain(sc,
          transformed,
          params,
          model,
          output)
      }
    }
    model
  }

  def sgdTrain(sc : SparkContext,
               input : RDD[Example],
               params : AdditiveTrainerParams,
               model : AdditiveModel,
               output : String) : AdditiveModel = {
    val modelBC = sc.broadcast(model)
    val paramsBC = sc.broadcast(params)
    input
      .sample(false, params.subsample)
      .coalesce(params.numBags, true)
      .mapPartitions(partition => sgdPartition(partition, modelBC, paramsBC))
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

    TrainingUtils.saveModel(model, output)
    return model
  }

  def sgdMultiscaleTrain(sc : SparkContext,
                         input : RDD[Example],
                         params :AdditiveTrainerParams,
                         model : AdditiveModel,
                         output : String) : AdditiveModel = {
    val modelBC = sc.broadcast(model)
    val paramsBC = sc.broadcast(params)
    input
      .sample(false, params.subsample)
      .coalesce(params.numBags, true)
      .mapPartitionsWithIndex((index, partition) =>
                                sgdPartitionMultiscale(index, partition, modelBC, paramsBC))
      .groupByKey()
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
    TrainingUtils.saveModel(model, output)
    return model
  }

  def sgdPartition(partition : Iterator[Example],
                   modelBC : Broadcast[AdditiveModel],
                   paramsBC : Broadcast[AdditiveTrainerParams]): Iterator[((String, String), Function)] = {
    val workingModel = modelBC.value
    val params = paramsBC.value
    val output = sgdPartitionInternal(partition, workingModel, params)
    output.iterator
  }

  def sgdPartitionMultiscale(index : Int,
                              partition : Iterator[Example],
                              modelBC : Broadcast[AdditiveModel],
                              paramsBC : Broadcast[AdditiveTrainerParams]): Iterator[((String, String), Function)] = {
    val workingModel = modelBC.value
    val params = paramsBC.value
    val multiscale = params.multiscale
    val newBins = multiscale(index % multiscale.length)

    log.info("Resampling to %d bins".format(newBins))
    workingModel
      .getWeights
      .foreach(family => {
        family._2.foreach(feature => {
          feature._2.resample(newBins)
      })
    })

    val output = sgdPartitionInternal(partition, workingModel, params)
    output.iterator
  }

  private def aggregateFuncWeights(input: Iterable[Function],
                                   scale: Float,
                                   numBins: Int,
                                   smoothingTolerance: Float) : Function = {
    val head : Function = input.head
    val output = head.aggregate(input.asJava, scale, numBins)
    if (output.isInstanceOf[Spline]) {
      smoothSpline(smoothingTolerance, output.asInstanceOf[Spline])
    }
    output
  }

  private def sgdPartitionInternal(partition : Iterator[Example],
                                   workingModel : AdditiveModel,
                                   params : AdditiveTrainerParams) :
  HashMap[(String, String), Function] = {
    @volatile var lossSum : Double = 0.0
    @volatile var lossCount : Int = 0
    partition.foreach(example => {
        lossSum += pointwiseLoss(example.example.get(0), workingModel, params.loss, params)
        lossCount = lossCount + 1
        if (lossCount % params.lossMod == 0) {
          log.info("Loss = %f, samples = %d".format(lossSum / params.lossMod.toDouble, lossCount))
          lossSum = 0.0
        }
    })
    val output = HashMap[(String, String), Function]()
    workingModel
      .getWeights
      .foreach(family => {
      family._2.foreach(feature => {
        output.put((family._1, feature._1), feature._2)
      })
    })
    output
  }

  def pointwiseLoss(fv : FeatureVector,
                    workingModel : AdditiveModel,
                    loss : String,
                    params : AdditiveTrainerParams) : Double = {
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
  def updateLogistic(model : AdditiveModel,
                     fv : FeatureVector,
                     label : Double,
                     params : AdditiveTrainerParams) : Double = {
    val flatFeatures = Util.flattenFeatureWithDropout(fv, params.dropout)
    val prediction = model.scoreFlatFeatures(flatFeatures) / (1.0 - params.dropout)
    // To prevent blowup.
    val corr = scala.math.min(10.0, label * prediction)
    val expCorr = scala.math.exp(corr)
    val loss = scala.math.log(1.0 + 1.0 / expCorr)
    val grad = -label / (1.0 + expCorr)
    model.update(grad.toFloat,
                 params.learningRate.toFloat,
                 params.linfinityCap.toFloat,
                 flatFeatures)
    return loss
  }

  def updateHinge(model : AdditiveModel,
                  fv : FeatureVector,
                  label : Double,
                  params : AdditiveTrainerParams) : Double = {
    val flatFeatures = Util.flattenFeatureWithDropout(fv, params.dropout)
    val prediction = model.scoreFlatFeatures(flatFeatures) / (1.0 - params.dropout)
    val loss = scala.math.max(0.0, params.margin - label * prediction)
    if (loss > 0.0) {
      val grad = -label
      model.update(grad.toFloat,
                   params.learningRate.toFloat,
                   params.linfinityCap.toFloat,
                   flatFeatures)
    }
    return loss
  }

  def updateRegressor(model: AdditiveModel,
                      fv: FeatureVector,
                      label: Double,
                      params : AdditiveTrainerParams) : Double = {
    val flatFeatures = Util.flattenFeatureWithDropout(fv, params.dropout)
    val prediction = model.scoreFlatFeatures(flatFeatures) / (1.0 - params.dropout)
    // absolute difference
    val loss = math.abs(prediction - label)
    if (prediction - label > params.epsilon) {
      model.update(1.0f, params.learningRate.toFloat,
                   params.linfinityCap.toFloat, flatFeatures)
    } else if (prediction - label < -params.epsilon) {
      model.update(-1.0f, params.learningRate.toFloat,
                   params.linfinityCap.toFloat, flatFeatures)
    }
    return loss
  }

  private def transformExamples(input: RDD[Example],
                        config: Config,
                        key: String,
                        params: AdditiveTrainerParams): RDD[Example] = {
    if (params.isRanking) {
      LinearRankerUtils.transformExamples(input, config, key)
    } else {
      LinearRankerUtils.makePointwiseFloat(input, config, key)
    }
  }

  private def modelInitialization(input: RDD[Example],
                                  params: AdditiveTrainerParams): AdditiveModel = {
    // add functions to additive model
    val initialModel = if(params.initModelPath == "") {
      None
    } else {
      TrainingUtils.loadScoreModel(params.initModelPath)
    }

    // sample examples to be used for model initialization
    val initExamples = input.sample(false, params.subsample)
    if(initialModel.isDefined) {
      val newModel = initialModel.get.asInstanceOf[AdditiveModel]
      initModel(params.minCount, params, initExamples, newModel, false)
      newModel
    } else {
      val newModel = new AdditiveModel()
      initModel(params.minCount, params, initExamples, newModel, true)
      setPrior(params.priors, newModel)
      newModel
    }
  }

  // Initializes the model
  private def initModel(minCount : Int,
                        params: AdditiveTrainerParams,
                        examples : RDD[Example],
                        model : AdditiveModel,
                        overwrite : Boolean) = {
    val linearFeatureFamilies = params.linearFeatureFamilies
    val priors = params.priors
    val minMax = TrainingUtils
      .getFeatureStatistics(minCount, examples)
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
      model.addFunction(featureFamily, featureName,
        new Linear(stats.min.toFloat, stats.min.toFloat), overwrite)
    }
  }

  def smoothSpline(tolerance : Double,
                   spline : Spline) = {
    val weights = spline.getWeights
    val optimize = weights.map(x => x.toDouble).toArray
    val errAndCoeff = SplineTrainer.fitPolynomial(optimize)
    if (errAndCoeff._1 < tolerance) {
      SplineTrainer.evaluatePolynomial(errAndCoeff._2, optimize, true)
      for (i <- 0 until weights.length) {
        weights(i) = optimize(i).toFloat
      }
    }
  }

  def deleteSmallFunctions(model : AdditiveModel,
                           linfinityThreshold : Double) = {
    val toDelete = scala.collection.mutable.ArrayBuffer[(String, String)]()

    model.getWeights.asScala.foreach(family => {
      family._2.asScala.foreach(entry => {
        val func : Function = entry._2
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
        val tokens : Array[String] = prior.split(",")
        if (tokens.length == 4) {
          val family = tokens(0)
          val name = tokens(1)
          val params = Array(tokens(2).toFloat, tokens(3).toFloat)
          val familyMap = model.getWeights.get(family)
          if (!familyMap.isEmpty) {
            val func : Function = familyMap.get(name)
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
      case _ : Throwable => log.info("No prior given")
    }
  }

  def loadTrainingParameters(config: Config): AdditiveTrainerParams = {
    val loss : String = config.getString("loss")
    val isRanking = loss match {
      case "logistic" => false
      case "hinge" => false
      case "regression" => false
      case _ => {
        log.error("Unknown loss function %s".format(loss))
        System.exit(-1)
        false
      }
    }
    val numBins : Int = config.getInt("num_bins")
    val numBags : Int = config.getInt("num_bags")
    val rankKey : String = config.getString("rank_key")
    val learningRate : Double = config.getDouble("learning_rate")
    val dropout : Double = config.getDouble("dropout")
    val subsample : Double = config.getDouble("subsample")
    val linfinityCap : Double = config.getDouble("linfinity_cap")
    val smoothingTolerance : Double = config.getDouble("smoothing_tolerance")
    val linfinityThreshold : Double = config.getDouble("linfinity_threshold")
    val initModelPath : String = Try{config.getString("init_model")}.getOrElse("")
    val threshold : Double = config.getDouble("rank_threshold")
    val epsilon: Double = Try{config.getDouble("epsilon")}.getOrElse(0.0)
    val minCount : Int = config.getInt("min_count")
    val linearFeatureFamilies : Array[String] = Try(
      config.getStringList("linear_feature").toList.toArray)
      .getOrElse(Array[String]())
    val lossMod : Int = Try{config.getInt("loss_mod")}.getOrElse(100)
    val priors : Array[String] = Try(
      config.getStringList("prior").toList.toArray)
    .getOrElse(Array[String]())

    val margin : Double = Try(config.getDouble("margin")).getOrElse(1.0)

    val multiscale : Array[Int] = Try(
      config.getIntList("multiscale").asScala.map(x => x.toInt).toArray)
      .getOrElse(Array[Int]())

    val rankMargin : Double = Try(config.getDouble("rank_margin")).getOrElse(0.5)

    AdditiveTrainerParams(
      numBins,
      numBags,
      rankKey,
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
      priors)
  }

  def trainAndSaveToFile(sc : SparkContext,
                         input : RDD[Example],
                         config : Config,
                         key : String) = {
    val model = train(sc, input, config, key)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }
}