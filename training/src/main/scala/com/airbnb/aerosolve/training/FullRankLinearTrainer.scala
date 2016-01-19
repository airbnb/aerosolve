package com.airbnb.aerosolve.training

import java.io.BufferedWriter
import java.io.OutputStreamWriter
import java.util.concurrent.ConcurrentHashMap

import com.airbnb.aerosolve.core.{ModelRecord, ModelHeader, FeatureVector, Example}
import com.airbnb.aerosolve.core.models.FullRankLinearModel
import com.airbnb.aerosolve.core.util.Util
import com.typesafe.config.Config
import org.slf4j.{LoggerFactory, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.util.Try
import scala.util.Random
import scala.math.abs
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path

/*
 * A trainer that generates a full rank linear model.
 *
 * */

object FullRankLinearTrainer {
  private final val log: Logger = LoggerFactory.getLogger("FullRankLinearTrainer")
  private final val lossKey = ("$loss", "$loss")
  val MAX_WEIGHTS : Int = 1000000
  
  case class FullRankLinearTrainerOptions(loss : String,
                                          iterations : Int,
                                          rankKey : String,
                                          learningRate : Double,
                                          subsample : Double,
                                          lambda : Double,
                                          lambda2 : Double)
  
  def parseTrainingOptions(config : Config) : FullRankLinearTrainerOptions = {
 
    FullRankLinearTrainerOptions(
        loss = config.getString("loss"),
        iterations = config.getInt("iterations"),
        rankKey = config.getString("rank_key"),
        learningRate = config.getDouble("learning_rate"),
        subsample = config.getDouble("subsample"),
        lambda = config.getDouble("lambda"),
        lambda2 = config.getDouble("lambda2")       
    )    
  }
  
  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String) : FullRankLinearModel = {
    val options = parseTrainingOptions(config.getConfig(key))

    val pointwise : RDD[Example] =
      LinearRankerUtils
        .makePointwiseFloat(input, config, key)
        .cache()
        
    val model = new FullRankLinearModel()
    model
  }
  def trainAndSaveToFile(sc : SparkContext,
                         input : RDD[Example],
                         config : Config,
                         key : String) = {
    val model = train(sc, input, config, key)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }
}
