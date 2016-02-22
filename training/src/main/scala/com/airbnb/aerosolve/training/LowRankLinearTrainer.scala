package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core.{Example, LabelDictionaryEntry}
import com.airbnb.aerosolve.core.models.LowRankLinearModel
import com.airbnb.aerosolve.core.util.FloatVector
import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.training.GradientUtils._
import com.typesafe.config.Config
import org.slf4j.{LoggerFactory, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import scala.collection.JavaConversions._
import scala.util.{Try, Random}

/*
 * A trainer that generates a low rank linear model.
 *
 * */

object LowRankLinearTrainer {
  private final val log: Logger = LoggerFactory.getLogger("LowRankLinearTrainer")
  private final val LABEL_EMBEDDING_KEY = "$label_embedding"
  case class LowRankLinearTrainerOptions(loss : String,
                                         iterations : Int,
                                         rankKey : String,
                                         lambda : Double,
                                         subsample : Double,
                                         minCount : Int,
                                         cache : String,
                                         solver : String,
                                         embeddingDimension : Int,
                                         rankLossType: String,
                                         maxNorm: Double)

  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String) : LowRankLinearModel = {
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

    options.cache match {
      case "memory" => pointwise.unpersist()
    }
    model
  }

  def modelIteration(sc : SparkContext,
                     options : LowRankLinearTrainerOptions,
                     model : LowRankLinearModel,
                     pointwise : RDD[Example]) = {
    var prevGradients : Map[(String, String), GradientContainer] = Map()
    val step = scala.collection.mutable.HashMap[(String, String), FloatVector]()
    for (iter <- 0 until options.iterations) {
      log.info(s"Iteration $iter")
      val sample = pointwise
        .sample(false, options.subsample)
      val gradients: Map[(String, String), GradientContainer] = options.loss match {
        case "hinge" => hingeGradient(sc, options ,model, sample)
        case _: String => hingeGradient(sc, options, model, sample)
      }

      val featureWeightVector = model.getFeatureWeightVector
      val labelWeightVector = model.getLabelWeightVector
      val labelWeightVectorWrapper =  new java.util.HashMap[String,java.util.Map[String,com.airbnb.aerosolve.core.util.FloatVector]]()
      labelWeightVectorWrapper.put(LABEL_EMBEDDING_KEY, labelWeightVector)
      options.solver match {
        // TODO (Peng): implement alternating optimization with bagging
        case "rprop" => {
          GradientUtils.rprop(gradients, prevGradients, step, featureWeightVector, options.embeddingDimension, options.lambda)
          GradientUtils.rprop(gradients, prevGradients, step, labelWeightVectorWrapper, options.embeddingDimension, options.lambda)
          prevGradients = gradients
          normalizeWeightVectors(model, options.maxNorm)
        }
      }
    }
  }

  def hingeGradient(sc : SparkContext,
                    options : LowRankLinearTrainerOptions,
                    model : LowRankLinearModel,
                    pointwise : RDD[Example]) : Map[(String, String), GradientContainer] = {
    val modelBC = sc.broadcast(model)

    pointwise
      .mapPartitions(partition => {
        val model = modelBC.value
        val labelToIdx = model.getLabelToIndex
        val dim = model.getLabelDictionary.size()
        val gradient = scala.collection.mutable.HashMap[(String, String), GradientContainer]()
        val featureWeightVector = model.getFeatureWeightVector
        val labelWeightVector = model.getLabelWeightVector
        val rnd = new Random()

        partition.foreach(examples => {
          val flatFeatures = Util.flattenFeature(examples.example.get(0))
          val scores = model.scoreFlatFeature(flatFeatures)
          val labels = flatFeatures.get(options.rankKey)

          if (labels != null && labels.size() > 0) {
            val posLabels = labels.toArray
            // Pick a random positive label
            val posLabelRnd = rnd.nextInt(posLabels.length)
            val (posLabel, posMargin) = posLabels(posLabelRnd)
            val posIdx = labelToIdx.get(posLabel)
            val posScore = scores.values(posIdx)
            var negScore = 0.0
            var negIdx = 0
            var negLabel = ""
            var negMargin = 0.0
            var N = 0
            do {
              // Pick a random other label
              val (idx, label, margin, iter) = pickRandomOtherLabel(model, labels, posIdx, posMargin, rnd, dim)
              if (iter < dim) {
                // we successfully get a random other label
                negScore = scores.values(negIdx)
                negMargin = margin
                negIdx = idx
                negLabel = label
                N += 1
              } else {
                // if we cannot find a random other label that has smaller margin compared to posMargin
                // we skip learning for the current example
                N = dim
              }
              // break the loop if the random other label is violating the margin requirement
            } while (N < dim && negScore - posScore <= negMargin - posMargin)

            if (N < dim) {
              // loss = max(0, margin + w(-)' * V * x - w(+)' * V * x) rankLoss
              // d(loss)/d(w-) =  rankLoss * V * x
              // d(loss)/d(w+) = - rankLoss * V * x
              // d(loss)/d(v) = rankLoss * w(-) * x - rankLoss * w(+) * x
              val rankLoss = rankToLoss(Math.floor((dim - 1) / N).toInt, dim, options.rankLossType)
              val loss = ((posMargin - negMargin) + (negScore - posScore)) * rankLoss
              if (loss > 0.0) {
                // compute gradient w.r.t W (labelWeightVector)
                val fvProjection = model.projectFeatureToEmbedding(flatFeatures)
                // update w-
                val negLabelKey = (LABEL_EMBEDDING_KEY, negLabel)
                val gradContainerNeg = gradient.getOrElse(negLabelKey,
                  GradientContainer(new FloatVector(options.embeddingDimension), 0.0))
                gradContainerNeg.grad.multiplyAdd(rankLoss, fvProjection)
                // the real square sum can be computed by:
                // featureSquaredSum += Math.max(fvProjection.dot(fvProjection), 1.0)
                // but with rprop solver, this is not used, so we don't compute it here to improve speed
                gradient.put(negLabelKey, GradientContainer(gradContainerNeg.grad, 0.0))
                // update w+
                val posLabelKey = (LABEL_EMBEDDING_KEY, posLabel)
                val gradContainerPos = gradient.getOrElse(posLabelKey,
                  GradientContainer(new FloatVector(options.embeddingDimension), 0.0))
                gradContainerPos.grad.multiplyAdd(-rankLoss, fvProjection)
                gradient.put(posLabelKey, GradientContainer(gradContainerPos.grad, 0.0))

                // compute gradient w.r.t V (featureWeightVector)
                val posLabelWeightVector = labelWeightVector.get(posLabel)
                val negLabelWeightVector = labelWeightVector.get(negLabel)
                for (family <- flatFeatures) {
                  for (feature <- family._2) {
                    val key = (family._1, feature._1)
                    // We only care about features in the model.
                    if (featureWeightVector.containsKey(key._1) && featureWeightVector.get(key._1).containsKey(key._2)) {
                      val featureVal = feature._2
                      val gradContainer = gradient.getOrElse(key,
                        GradientContainer(new FloatVector(options.embeddingDimension), 0.0))
                      gradContainer.grad.multiplyAdd(featureVal.toFloat * rankLoss, negLabelWeightVector)
                      gradContainer.grad.multiplyAdd(-featureVal.toFloat * rankLoss, posLabelWeightVector)
                      gradient.put(key, GradientContainer(gradContainer.grad, 0.0))
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

  def trainAndSaveToFile(sc : SparkContext,
                         input : RDD[Example],
                         config : Config,
                         key : String) = {
    val model = train(sc, input, config, key)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }

  private def normalizeWeightVectors(model: LowRankLinearModel, maxNorm: Double) = {
    val featureWeightVector = model.getFeatureWeightVector
    val labelWeightVector = model.getLabelWeightVector
    for (family <- featureWeightVector.entrySet()) {
      for (feature <- family.getValue.entrySet()) {
        val weight = feature.getValue
        weight.capNorm(maxNorm.toFloat)
      }
    }

    for (labelWeight <- labelWeightVector.entrySet()) {
      val weight = labelWeight.getValue
      weight.capNorm(maxNorm.toFloat)
    }
  }

  private def parseTrainingOptions(config : Config) : LowRankLinearTrainerOptions = {
    LowRankLinearTrainerOptions(
      loss = config.getString("loss"),
      iterations = config.getInt("iterations"),
      rankKey = config.getString("rank_key"),
      lambda = config.getDouble("lambda"),
      subsample = config.getDouble("subsample"),
      minCount = config.getInt("min_count"),
      cache = Try(config.getString("cache")).getOrElse(""),
      solver = Try(config.getString("solver")).getOrElse("rprop"),
      embeddingDimension = config.getInt("embedding_dimension"),
      rankLossType = Try(config.getString("rank_loss")).getOrElse("non_uniform"),
      maxNorm = Try(config.getDouble("max_norm")).getOrElse(1.0)
    )
  }

  private def pickRandomOtherLabel(model: LowRankLinearModel,
                                   posLabels: java.util.Map[java.lang.String, java.lang.Double],
                                   posIdx: Int,
                                   posMargin: Double,
                                   rnd: Random,
                                   dim: Int) : (Int, String, Double, Int) = {
    // Pick a random other label. This can be a negative or a positive with a smaller margin.
    var negIdx = rnd.nextInt(dim)
    var negLabel = model.getLabelDictionary.get(negIdx).label
    var negMargin : Double = if (posLabels.containsKey(negLabel)) posLabels.get(negLabel) else 0.0
    var iter = 0
    while ((negIdx == posIdx || negMargin > posMargin) && iter < dim) {
      // we only want to pick a label that has smaller margin, we try at most dim times
      // we try this for at most dim times
      negIdx = rnd.nextInt(dim)
      negLabel = model.getLabelDictionary.get(negIdx).label
      negMargin = if (posLabels.containsKey(negLabel)) posLabels.get(negLabel) else 0.0
      iter += 1
    }
    (negIdx, negLabel, negMargin, iter)
  }

  private def rankToLoss(k : Int, dim: Int, lossType: String): Float = {
    lossType match {
      case "uniform" => {
        uniformRankLoss(k, dim)
      }
      case "non_uniform" => {
        nonUniformRankLoss(k, dim)
      }
      case _ => {
        1.0f
      }
    }
  }

  private def uniformRankLoss(k: Int, dim: Int): Float = {
    // dim is the number of classes
    // assume we always have dim > 1
    k * 1.0f / (dim - 1.0f)
  }

  private def nonUniformRankLoss(k: Int, dim: Int): Float = {
    // (TODO) peng: make this an array for efficiency
    var loss = 0.0f
    for (i <- 1 to k) {
      loss += 1.0f / i
    }
    loss
  }

  def setupModel(options : LowRankLinearTrainerOptions, pointwise : RDD[Example]) : LowRankLinearModel = {
    val stats = TrainingUtils.getFeatureStatistics(options.minCount, pointwise)
    val model = new LowRankLinearModel()
    val featureWeights = model.getFeatureWeightVector
    val labelWeights = model.getLabelWeightVector
    val dict = model.getLabelDictionary
    val embeddingSize = options.embeddingDimension
    model.setEmbeddingDimension(embeddingSize)
    var count : Int = 0
    for (kv <- stats) {
      val (family, feature) = kv._1
      if (family == options.rankKey) {
        val entry = new LabelDictionaryEntry()
        entry.setLabel(feature)
        entry.setCount(kv._2.count.toInt)
        dict.add(entry)
      } else {
        if (!featureWeights.containsKey(family)) {
          featureWeights.put(family, new java.util.HashMap[java.lang.String, FloatVector]())
        }
        val familyMap = featureWeights.get(family)
        if (!familyMap.containsKey(feature)) {
          count = count + 1
          familyMap.put(feature, FloatVector.getUniformVector(embeddingSize))
        }
      }
    }

    for (labelEntry <- dict) {
      val labelName = labelEntry.getLabel
      val fv = FloatVector.getUniformVector(embeddingSize)
      labelWeights.put(labelName, fv)
    }

    model.buildLabelToIndex()
    normalizeWeightVectors(model, options.maxNorm)
    log.info(s"Total number of labels is ${dict.size()}")
    log.info(s"Total number of features is $count")

    model
  }
}
