package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core.{Example, FeatureVector, FunctionForm}
import com.airbnb.aerosolve.core.features.{MultiFamilyVector, Family, FeatureRegistry}
import com.airbnb.aerosolve.core.models.KernelModel
import com.airbnb.aerosolve.core.util.{FloatVector, SupportVector}
import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.util.Try

// Simple SGD based kernel trainer. Mostly so we can test the kernel model for online use.
// TODO(hector_yee) : if this gets more heavily used add in regularization and better training.
object KernelTrainer {
  val log: Logger = LoggerFactory.getLogger("KernelTrainer")
  
  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String,
            registry: FeatureRegistry) : KernelModel = {
    val modelConfig = config.getConfig(key)
    val candidateSize : Int = modelConfig.getInt("num_candidates")
    val kernel : String = modelConfig.getString("kernel")
    val maxSV : Int = modelConfig.getInt("max_vectors")
    val scale : Float = modelConfig.getDouble("scale").toFloat

    val learningRate : Double = Try(modelConfig.getDouble("learning_rate")).getOrElse(0.1d)
    val labelFamily : Family = registry.family(modelConfig.getString("rank_key"))
    val rankThreshold : Double = Try(modelConfig.getDouble("rank_threshold")).getOrElse(0.0d)

    val examples = LinearRankerUtils
        .makePointwiseFloat(input, config, key, registry)
    val model = initModel(modelConfig, examples, registry)
        
    val loss : String = modelConfig.getString("loss")
        
    val candidates = examples.take(candidateSize)

    // Super simple SGD trainer. Mostly to get the unit test to pass
    for (candidate <- candidates) {
      val gradient = computeGradient(model, candidate.only, loss, labelFamily, rankThreshold)
      if (gradient != 0.0) {
        val vector = candidate.only
        val vec = model.dictionary.makeVectorFromSparseFloats(vector)
        addNewSupportVector(model, kernel, scale, vec, maxSV)
        model.onlineUpdate(gradient, learningRate, vector)
      }
    }
   
    model
  }

  def addNewSupportVector(model : KernelModel, kernel : String, scale : Float, vec : FloatVector, maxSV : Int) = {
    val supportVectors = model.supportVectors
    if (supportVectors.size() < maxSV) {
      val form = kernel match {
        case "rbf" => FunctionForm.RADIAL_BASIS_FUNCTION
        case "acos" => FunctionForm.ARC_COSINE
        case "random" => scala.util.Random.nextInt(2) match {
          case 0 => FunctionForm.RADIAL_BASIS_FUNCTION
          case 1 => FunctionForm.ARC_COSINE
        }
      }
      val sv = new SupportVector(vec, form, scale, 0.0f)
      supportVectors.add(sv)
    }
  }

  def computeGradient(model : KernelModel,
                      fv : MultiFamilyVector,
                      loss : String, labelFamily : Family, rankThreshold : Double) : Double = {
    val prediction = model.scoreItem(fv)
    val label = if (loss == "hinge") {
      TrainingUtils.getLabel(fv, labelFamily, rankThreshold)
    } else {
      TrainingUtils.getLabel(fv, labelFamily)
    }

    loss match {
      case "hinge" => {
        val lossVal = scala.math.max(0.0, 1.0 - label * prediction)
        if (lossVal > 0.0) {
           return -label.toFloat
        }
      }
      case "regression" => {
        val diff = prediction - label
        if (diff > 1.0) {
          return 1.0d
        }
        if (diff < -1.0) {
          return -1.0d
        }
      }
    }
    0.0d
  }

  def initModel(modelConfig : Config, examples : RDD[Example], registry: FeatureRegistry) : KernelModel = {
    val minCount : Int = modelConfig.getInt("min_count")
    val labelFamily : Family = registry.family(modelConfig.getString("rank_key"))
    
    log.info("Building dictionary")
    val stats = TrainingUtils.getFeatureStatistics(minCount, examples)
    log.info(s"Dictionary size is ${stats.size}")
    val dictionary = TrainingUtils.createStringDictionaryFromFeatureStatistics(stats, Set(labelFamily))
    
    val model = new KernelModel(registry)
    model.dictionary(dictionary)
    
    model
  }

  def trainAndSaveToFile(sc : SparkContext,
                         input : RDD[Example],
                         config : Config,
                         key : String,
                         registry: FeatureRegistry) = {
    val model = train(sc, input, config, key, registry)
    TrainingUtils.saveModel(model, config, key + "model_output")
  }
}
