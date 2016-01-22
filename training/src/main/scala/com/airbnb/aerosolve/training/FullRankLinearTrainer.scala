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

    for (iter <- 0 until options.iterations) {
      log.info(s"Iteration $iter")
      modelIteration(sc, options, model, pointwise)
    }
    model
  }

  def trainAndSaveToFile(sc : SparkContext,
                         input : RDD[Example],
                         config : Config,
                         key : String) = {
    val model = train(sc, input, config, key)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }
  
  def modelIteration(sc : SparkContext,
                     options : FullRankLinearTrainerOptions,
                     model : FullRankLinearModel,
                     pointwise : RDD[Example]) = {
    val gradient : Array[((String, String), FloatVector)] = options.loss match {
      case "softmax" => softmaxGradient(sc, options, model, pointwise)
      case _ : String => softmaxGradient(sc, options, model, pointwise)
    }
  }
  
  def softmaxGradient(sc : SparkContext,
                      options : FullRankLinearTrainerOptions,
                      model : FullRankLinearModel,
                      pointwise : RDD[Example]) : Array[((String, String), FloatVector)] = {
    val modelBC = sc.broadcast(model)
    
    pointwise
    .mapPartitions(partition => {
      val model = modelBC.value
      val labelToIdx = model.getLabelToIndex()
      val dim = model.getLabelDictionary.size()
      val gradient = scala.collection.mutable.HashMap[(String, String), FloatVector]()
      val weightVector = model.getWeightVector()

      partition.foreach(examples => {
        val flatFeatures = Util.flattenFeature(examples.example.get(0))
        val labels = flatFeatures.get(options.rankKey)
        if (labels != null) {
          val posLabels = labels.keySet().asScala
          val scores = model.scoreFlatFeature(flatFeatures)
          // Convert to multinomial using softmax.
          scores.softmax()
          // The importance is prob - 1 for positive labels, prob otherwise.
          for (posLabel <- posLabels) {
            val posIdx = labelToIdx.get(posLabel)
            if (posIdx != null) {
              scores.values(posIdx) -= 1
            }
          }
          // Gradient is importance * feature value
          for (family <- flatFeatures) {
            for (feature <- family._2) {
              val key = (family._1, feature._1)
              // We only care about features in the model.
              if (weightVector.containsKey(key._1) && weightVector.get(key._1).containsKey(key._2)) {
                val featureVal = feature._2
                val grad = gradient.getOrElse(key, new FloatVector(dim))
                grad.multiplyAdd(featureVal.toFloat, scores)
                gradient.put(key, grad)
              }
            }
          }
        }
      })
      gradient.iterator
    })
    .reduceByKey((a, b) => {a.add(b); a})
    .collect
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
    model.buildLabelToIndex();

    model
  }
}
