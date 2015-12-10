package com.airbnb.aerosolve.training

import java.util

import com.airbnb.aerosolve.core.models.KernelModel
import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.core.FeatureVector
import com.airbnb.aerosolve.core.FunctionForm
import com.airbnb.aerosolve.core.ModelRecord
import com.airbnb.aerosolve.core.util.FloatVector
import com.airbnb.aerosolve.core.util.StringDictionary
import com.airbnb.aerosolve.core.util.SupportVector
import com.airbnb.aerosolve.core.util.Util
import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.util.Random
import scala.util.Try
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

// Simple SGD based kernel trainer. Mostly so we can test the kernel model for online use.
// TODO(hector_yee) : if this gets more heavily used add in regularization and better training.
object KernelTrainer {
  val log: Logger = LoggerFactory.getLogger("KernelTrainer")
  
  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String) : KernelModel = {
    val modelConfig = config.getConfig(key)
    val candidateSize : Int = modelConfig.getInt("num_candidates")
    val kernel : String = modelConfig.getString("kernel")
    val maxSV : Int = modelConfig.getInt("max_vectors")
    val scale : Float = modelConfig.getDouble("scale").toFloat

    val learningRate : Float = Try(modelConfig.getDouble("learning_rate").toFloat).getOrElse(0.1f)
    val rankKey : String = modelConfig.getString("rank_key")
    val rankThreshold : Double = Try(modelConfig.getDouble("rank_threshold")).getOrElse(0.0f)

    val examples = LinearRankerUtils
        .makePointwiseFloat(input, config, key)
    val model = initModel(modelConfig, examples)
        
    val loss : String = modelConfig.getString("loss")
        
    val candidates = examples.take(candidateSize)

    // Super simple SGD trainer. Mostly to get the unit test to pass
    for (candidate <- candidates) {
      val gradient = computeGradient(model, candidate.example(0), loss, rankKey, rankThreshold)
      if (gradient != 0.0) {
        val flatFeatures = Util.flattenFeature(candidate.example(0));
        val vec = model.getDictionary().makeVectorFromSparseFloats(flatFeatures);
        addNewSupportVector(model, kernel, scale, vec, maxSV)
        model.onlineUpdate(gradient, learningRate, flatFeatures)
      }
    }
   
    model
  }

  def addNewSupportVector(model : KernelModel, kernel : String, scale : Float, vec : FloatVector, maxSV : Int) = {
    val supportVectors = model.getSupportVectors()
    if (supportVectors.size() < maxSV) {
      val form = kernel match {
        case "rbf" => FunctionForm.RADIAL_BASIS_FUNCTION
        case "acos" => FunctionForm.ARC_COSINE
        case "random" => scala.util.Random.nextInt(2) match {
          case 0 => FunctionForm.RADIAL_BASIS_FUNCTION
          case 1 => FunctionForm.ARC_COSINE
        }
      }
      val sv = new SupportVector(vec, form, scale, 0.0f);
      supportVectors.add(sv);
    }
  }

  def computeGradient(model : KernelModel,
                      fv : FeatureVector,
                      loss : String, rankKey : String, rankThreshold : Double) : Float = {
    val prediction = model.scoreItem(fv)
    val label = if (loss == "hinge") TrainingUtils.getLabel(fv, rankKey, rankThreshold) else TrainingUtils.getLabel(fv, rankKey)
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
          return 1.0f
        }
        if (diff < -1.0) {
          return -1.0f
        }
      }
    }
    return 0.0f;
  }

  def initModel(modelConfig : Config, examples : RDD[Example]) : KernelModel = {
    val minCount : Int = modelConfig.getInt("min_count")
    val rankKey : String = modelConfig.getString("rank_key")
    
    log.info("Building dictionary")
    val stats = TrainingUtils.getFeatureStatistics(minCount, examples)
    log.info(s"Dictionary size is ${stats.size}")
    val dictionary = TrainingUtils.createStringDictionaryFromFeatureStatistics(stats, Set(rankKey))
    
    val model = new KernelModel()
    model.setDictionary(dictionary)
    
    model
  }

  def trainAndSaveToFile(sc : SparkContext,
                         input : RDD[Example],
                         config : Config,
                         key : String) = {
    val model = train(sc, input, config, key)
    TrainingUtils.saveModel(model, config, key + "model_output")
  }
}
