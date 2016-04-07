package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core.{Example, LabelDictionaryEntry}
import com.airbnb.aerosolve.core.models.FullRankLinearModel
import com.airbnb.aerosolve.core.util.FloatVector
import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.training.GradientUtils._
import com.typesafe.config.Config
import org.slf4j.{LoggerFactory, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.util.{Try, Random}

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
                                          subsample : Double,
                                          minCount : Int,
                                          cache : String,
                                          solver : String,
                                          labelMinCount: Option[Int])

  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String) : FullRankLinearModel = {
    val options = parseTrainingOptions(config.getConfig(key))

    val raw : RDD[Example] =
      LinearRankerUtils
        .makePointwiseFloat(input, config, key)

    val pointwise = options.cache match {
      case "memory" => raw.cache()
      case _ : String => raw
    }

    val model = setupModel(options, pointwise)

    modelIteration(sc, options, model, pointwise)

    filterZeros(model)

    options.cache match {
      case "memory" => pointwise.unpersist()
      case _ : String => Unit
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
    var prevGradients : Map[(String, String), GradientContainer] = Map()
    val step = scala.collection.mutable.HashMap[(String, String), FloatVector]()
    for (iter <- 0 until options.iterations) {
      log.info(s"Iteration $iter")
      val sample = pointwise.sample(false, options.subsample)
      val gradients: Map[(String, String), GradientContainer] = options.loss match {
        case "softmax" => softmaxGradient(sc, options, model, sample)
        case "hinge" => hingeGradient(sc, options ,model, sample, "l1")
        case "squared_hinge" => hingeGradient(sc, options, model, sample, "l2")
        case _: String => softmaxGradient(sc, options, model, sample)
      }
      val weightVector = model.getWeightVector()
      val dim = model.getLabelDictionary.size()
      options.solver match {
        case "sparse_boost" => GradientUtils
          .sparseBoost(gradients, weightVector, dim, options.lambda)
        case "rprop" => {
          GradientUtils.rprop(gradients, prevGradients, step, weightVector, dim, options.lambda)
          prevGradients = gradients
        }
      }
    }
  }

  def filterZeros(model : FullRankLinearModel) = {
    val weightVector = model.getWeightVector()
    for (family <- weightVector) {
      val toDelete = scala.collection.mutable.ArrayBuffer[String]()
      for (feature <- family._2) {
        if (feature._2.dot(feature._2) < 1e-6) {
          toDelete.add(feature._1)
        }
      }
      for (deleteFeature <- toDelete) {
        family._2.remove(deleteFeature)
      }
    }
  }
  
  def softmaxGradient(sc : SparkContext,
                      options : FullRankLinearTrainerOptions,
                      model : FullRankLinearModel,
                      pointwise : RDD[Example]) : Map[(String, String), GradientContainer] = {
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
                val norm = math.max(featureVal * featureVal, 1.0)
                gradient.put(key,
                             GradientContainer(gradContainer.grad,
                             gradContainer.featureSquaredSum + norm
                ))
              }
            }
          }
        }
      })
      gradient.iterator
    })
    .reduceByKey((a, b) => GradientUtils.sumGradients(a,b))
    .collectAsMap
    .toMap
  }

  def hingeGradient(sc : SparkContext,
                    options : FullRankLinearTrainerOptions,
                    model : FullRankLinearModel,
                    pointwise : RDD[Example],
                    lossType : String) : Map[(String, String), GradientContainer] = {
    val modelBC = sc.broadcast(model)

    pointwise
      .mapPartitions(partition => {
        val model = modelBC.value
        val labelToIdx = model.getLabelToIndex()
        val dim = model.getLabelDictionary.size()
        val gradient = scala.collection.mutable.HashMap[(String, String), GradientContainer]()
        val weightVector = model.getWeightVector()
        val rnd = new Random()

        partition.foreach(examples => {
          val flatFeatures = Util.flattenFeature(examples.example.get(0))
          val labels = flatFeatures.get(options.rankKey)
          if (labels != null && labels.size() > 0) {
            val posLabels = labels.toArray
            // Pick a random positive label
            val posLabelRnd = rnd.nextInt(posLabels.size)
            val (posLabel, posMargin) = posLabels(posLabelRnd)
            val posIdx = labelToIdx.get(posLabel)
            // Pick a random other label. This can be a negative or a positive with a smaller margin.
            var negIdx = rnd.nextInt(dim)
            while (negIdx == posIdx) {
              negIdx = rnd.nextInt(dim)
            }
            val negLabel = model.getLabelDictionary.get(negIdx).label
            val negMargin : Double = if (labels.containsKey(negLabel)) labels.get(negLabel) else 0.0

            if (posMargin > negMargin) {
              val scores = model.scoreFlatFeature(flatFeatures)
              val posScore = scores.values(posIdx)
              val negScore = scores.values(negIdx)
              // loss = max(0, margin + w(-) * x - w(+) * x)
              // so dloss / dw(-) = x and dloss / dw(+) = -x for hinge loss
              // and dloss / dw(-) = loss * x for squared hinge loss
              val loss = (posMargin - negMargin) + (negScore - posScore)
              if (loss > 0.0) {
                val grad = new FloatVector(dim)
                if (lossType == "l1" ) {
                  grad.values(posIdx) = -1.0f
                  grad.values(negIdx) = 1.0f
                } else {
                  grad.values(posIdx) = -loss.toFloat
                  grad.values(negIdx) = loss.toFloat
                }

                for (family <- flatFeatures) {
                  for (feature <- family._2) {
                    val key = (family._1, feature._1)
                    // We only care about features in the model.
                    if (weightVector.containsKey(key._1) && weightVector.get(key._1).containsKey(key._2)) {
                      val featureVal = feature._2
                      val gradContainer = gradient.getOrElse(key,
                                                             GradientContainer(new FloatVector(dim), 0.0))
                      gradContainer.grad.multiplyAdd(featureVal.toFloat, grad)
                      val norm = math.max(featureVal * featureVal, 1.0)
                      gradient.put(key,
                                   GradientContainer(gradContainer.grad,
                                                     gradContainer.featureSquaredSum + norm
                                   ))
                    }
                  }
                }
              }
            }
          }
        })
        gradient.iterator
      })
      .reduceByKey((a, b) => GradientUtils.sumGradients(a,b))
      .collectAsMap
      .toMap
  }

  def parseTrainingOptions(config : Config) : FullRankLinearTrainerOptions = {

    FullRankLinearTrainerOptions(
        loss = config.getString("loss"),
        iterations = config.getInt("iterations"),
        rankKey = config.getString("rank_key"),
        lambda = config.getDouble("lambda"),
        subsample = config.getDouble("subsample"),
        minCount = config.getInt("min_count"),
        cache = Try(config.getString("cache")).getOrElse(""),
        solver = Try(config.getString("solver")).getOrElse("rprop"),
        labelMinCount = Try(Some(config.getInt("label_min_count"))).getOrElse(None)
    )
  }
  
  def setupModel(options : FullRankLinearTrainerOptions, pointwise : RDD[Example]) : FullRankLinearModel = {
    val stats = TrainingUtils.getFeatureStatistics(options.minCount, pointwise)
    val labelCounts = if (options.labelMinCount.isDefined) {
      TrainingUtils.getLabelCounts(options.labelMinCount.get, pointwise, options.rankKey)
    } else {
      TrainingUtils.getLabelCounts(options.minCount, pointwise, options.rankKey)
    }

    val model = new FullRankLinearModel()
    val weights = model.getWeightVector()
    val dict = model.getLabelDictionary()

    for (kv <- stats) {
      val (family, feature) = kv._1
      if (family != options.rankKey) {
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

    for (kv <- labelCounts) {
      val (family, feature) = kv._1
      val entry = new LabelDictionaryEntry()
      entry.setLabel(feature)
      entry.setCount(kv._2)
      dict.add(entry)
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
    model.buildLabelToIndex()

    model
  }
}
