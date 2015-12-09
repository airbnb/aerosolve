package com.airbnb.aerosolve.training

import java.util

import com.airbnb.aerosolve.core.models.KernelModel
import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.core.FunctionForm
import com.airbnb.aerosolve.core.ModelRecord
import com.airbnb.aerosolve.core.util.StringDictionary
import com.airbnb.aerosolve.core.util.Util
import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.util.Random
import scala.util.Try
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

// The decision tree is meant to be a prior for the spline model / linear model
object KernelTrainer {
  val log: Logger = LoggerFactory.getLogger("KernelTrainer")

  // Mapping from names to kernel types
  private final val FunctionFormMap = Map(
    "rbf" -> FunctionForm.RADIAL_BASIS_FUNCTION,
    "acos" -> FunctionForm.ARC_COSINE
  )
  
  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String) : KernelModel = {
    val modelConfig = config.getConfig(key)
    val candidateSize : Int = modelConfig.getInt("num_candidates")
    val rankKey : String = modelConfig.getString("rank_key")
    val rankThreshold : Double = modelConfig.getDouble("rank_threshold")

    val examples = LinearRankerUtils
        .makePointwiseFloat(input, config, key)
    val model = initModel(modelConfig, examples)
        
    val loss : String = modelConfig.getString("loss")
        
    val candidates = examples    
        .map(x => Util.flattenFeature(x.example(0)))
        .filter(x => x.contains(rankKey))
        .take(candidateSize)
   

    model
  }
  
  def initModel(modelConfig : Config, examples : RDD[Example]) : KernelModel = {
    val kernel : String = modelConfig.getString("kernel")
    val scale : Double = modelConfig.getDouble("scale")
    val maxSV : Double = modelConfig.getDouble("max_vectors")
    val minDistance : Double = modelConfig.getDouble("min_distance")
    val minCount : Int = modelConfig.getInt("min_count")
    
    log.info("Building dictionary")
    val dictionary = new StringDictionary();
    val minMax : Array[((String, String), (Double, Double))] = TrainingUtils.getMinMax(minCount, examples)
    log.info(s"Dictionary size is ${minMax.size}")
    
    val model = new KernelModel()
    
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
