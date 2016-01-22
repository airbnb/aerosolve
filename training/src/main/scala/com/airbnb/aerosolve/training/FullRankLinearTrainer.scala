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
                                          lambda : Double,
                                          minCount : Int)

  case class GradientContainer(grad : FloatVector, featureSquaredSum : Double)

  def sumGradients(a : GradientContainer, b : GradientContainer) : GradientContainer = {
    a.grad.add(b.grad)
    GradientContainer(a.grad, a.featureSquaredSum + b.featureSquaredSum)
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
    val gradients : Array[((String, String), GradientContainer)] = options.loss match {
      case "softmax" => softmaxGradient(sc, options, model, pointwise)
      case _ : String => softmaxGradient(sc, options, model, pointwise)
    }
    val weightVector = model.getWeightVector()
    val dim = model.getLabelDictionary.size()
    var gradientNorm = 0.0
    var featureCount = 0
    // Gradient update rule from "boosting with structural sparsity Duchi et al 2009"
    gradients.foreach(kv => {
      val (key, gradient) = kv
      val featureMap = weightVector.get(key._1)
      if (featureMap != null) {
        val weight = featureMap.get(key._2)
        if (weight != null) {
          // Just a proxy measure for convergence.
          gradientNorm = gradientNorm + gradient.grad.dot(gradient.grad)
          val scale = 2.0 / math.max(1e-6, gradient.featureSquaredSum)
          weight.multiplyAdd(-scale.toFloat, gradient.grad)
          val hingeScale = 1.0 - options.lambda * scale / math.sqrt(weight.dot(weight))
          if (hingeScale <= 0.0f) {
            // This entire weight got regularized away.
            featureMap.remove(key._2)
          } else {
            featureCount = featureCount + 1
            weight.scale(hingeScale.toFloat)
          }
        }
      }
    })
    log.info("Sum of Gradient L2 norms = " + gradientNorm)
    log.info("Num active features = " + featureCount)
  }
  
  def softmaxGradient(sc : SparkContext,
                      options : FullRankLinearTrainerOptions,
                      model : FullRankLinearModel,
                      pointwise : RDD[Example]) : Array[((String, String), GradientContainer)] = {
    val modelBC = sc.broadcast(model)
    
    pointwise
    .mapPartitions(partition => {
      val model = modelBC.value
      val labelToIdx = model.getLabelToIndex()
      val dim = model.getLabelDictionary.size()
      val gradient = scala.collection.mutable.HashMap[(String, String), GradientContainer]()
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
                val gradContainer = gradient.getOrElse(key,
                                                       GradientContainer(new FloatVector(dim), 0.0))
                gradContainer.grad.multiplyAdd(featureVal.toFloat, scores)
                gradient.put(key,
                             GradientContainer(gradContainer.grad,
                             gradContainer.featureSquaredSum + featureVal * featureVal
                ))
              }
            }
          }
        }
      })
      gradient.iterator
    })
    .reduceByKey((a, b) => sumGradients(a,b))
    .collect
  }

  def parseTrainingOptions(config : Config) : FullRankLinearTrainerOptions = {
 
    FullRankLinearTrainerOptions(
        loss = config.getString("loss"),
        iterations = config.getInt("iterations"),
        rankKey = config.getString("rank_key"),
        lambda = config.getDouble("lambda"),
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
