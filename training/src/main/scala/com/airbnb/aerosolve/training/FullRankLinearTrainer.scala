package com.airbnb.aerosolve.training

import java.io.BufferedWriter
import java.io.OutputStreamWriter
import java.util.concurrent.ConcurrentHashMap

import com.airbnb.aerosolve.core.{ModelRecord, ModelHeader, FeatureVector, Example, LabelDictionaryEntry}
import com.airbnb.aerosolve.core.models.FullRankLinearModel
import com.airbnb.aerosolve.core.util.FloatVector
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
  
  case class FullRankLinearTrainerOptions(loss : String,
                                          iterations : Int,
                                          rankKey : String,
                                          learningRate : Double,
                                          subsample : Double,
                                          lambda : Double,
                                          lambda2 : Double,
                                          minCount : Int)
  
  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String) : FullRankLinearModel = {
    val options = parseTrainingOptions(config.getConfig(key))

    val pointwise : RDD[Example] =
      LinearRankerUtils
        .makePointwiseFloat(input, config, key)

    val model = setupModel(options, pointwise)
    model
  }

  def trainAndSaveToFile(sc : SparkContext,
                         input : RDD[Example],
                         config : Config,
                         key : String) = {
    val model = train(sc, input, config, key)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }

  def parseTrainingOptions(config : Config) : FullRankLinearTrainerOptions = {
 
    FullRankLinearTrainerOptions(
        loss = config.getString("loss"),
        iterations = config.getInt("iterations"),
        rankKey = config.getString("rank_key"),
        learningRate = config.getDouble("learning_rate"),
        subsample = config.getDouble("subsample"),
        lambda = config.getDouble("lambda"),
        lambda2 = config.getDouble("lambda2"),
        minCount = config.getInt("min_count")
    )    
  }
  
  def setupModel(options : FullRankLinearTrainerOptions, pointwise : RDD[Example]) : FullRankLinearModel = {
    val stats = TrainingUtils.getFeatureStatistics(options.minCount, pointwise)
    val model = new FullRankLinearModel()
    val weights = model.getWeightVector()
    val dict = model.getLabelDictionary()

    for (kv <- stats) {
      val (family, feature) = kv._1
      if (family == options.rankKey) {
        val entry = new LabelDictionaryEntry()
        entry.setLabel(feature)
        entry.setCount(kv._2.count.toInt)
        dict.add(entry)
      } else {
        if (!weights.containsKey(family)) {
          weights.put(family, new java.util.HashMap[java.lang.String, FloatVector]())
        }
        val familyMap = weights.get(family)
        if (!familyMap.containsKey(feature)) {
          // Dummy entry until we know the number of labels.
          familyMap.put(feature, null)
        }
      }
    }

    val dim = dict.size()
    log.info(s"Total number of labels is $dim")

    // Now fill all the feature vectors with length dim.
    var count : Int = 0
    for (family <- weights) {
      val keys = family._2.keySet()
      for (key <- keys) {
        count = count + 1
        family._2.put(key, new FloatVector(dim))
      }
    }
    log.info(s"Total number of features is $count")

    model
  }
}
