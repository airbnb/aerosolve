package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core.features.{Family, Feature, FeatureRegistry}
import com.airbnb.aerosolve.core.models.LowRankLinearModel
import com.airbnb.aerosolve.core.util.FloatVector
import com.airbnb.aerosolve.core.{Example, LabelDictionaryEntry}
import com.airbnb.aerosolve.training.GradientUtils._
import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.util.{Random, Try}

/*
 * A trainer that generates a low rank linear model.
 *
 * */

object LowRankLinearTrainer {
  private final val log: Logger = LoggerFactory.getLogger("LowRankLinearTrainer")
  private final val LABEL_EMBEDDING_KEY = "$label_embedding"
  case class LowRankLinearTrainerOptions(loss : String,
                                         iterations : Int,
                                         labelFamily : Family,
                                         lambda : Double,
                                         subsample : Double,
                                         minCount : Int,
                                         cache : String,
                                         solver : String,
                                         embeddingDimension : Int,
                                         rankLossType: String,
                                         maxNorm: Double,
                                         registry: FeatureRegistry)

  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String,
            registry: FeatureRegistry) : LowRankLinearModel = {
    val options = parseTrainingOptions(config.getConfig(key), registry)

    val raw : RDD[Example] =
      LinearRankerUtils
        .makePointwiseFloat(input, config, key, options.registry)

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
    var prevGradients : Map[Feature, GradientContainer] = Map()
    val step = scala.collection.mutable.HashMap[Feature, FloatVector]()
    for (iter <- 0 until options.iterations) {
      log.info(s"Iteration $iter")
      val sample = pointwise
        .sample(false, options.subsample)
      val gradients: Map[Feature, GradientContainer] = options.loss match {
        case "hinge" => hingeGradient(sc, options ,model, sample)
        case _: String => hingeGradient(sc, options, model, sample)
      }

      val featureWeightVector = model.featureWeightVector
      val labelWeightVector = model.labelWeightVector()
      val labelWeightVectorWrapper =  labelWeightVector.map{ case (featureName, vector) =>
        (options.registry.feature(LABEL_EMBEDDING_KEY, featureName), vector)
      }.toMap

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
                    pointwise : RDD[Example]) : Map[Feature, GradientContainer] = {
    val modelBC = sc.broadcast(model)

    pointwise
      .mapPartitions(partition => {
        val model = modelBC.value
        val labelToIdx = model.labelToIndex
        val dim = model.labelDictionary.size
        val gradient = mutable.HashMap[Feature, GradientContainer]()
        val featureWeightVector = model.featureWeightVector
        val labelWeightVector = model.labelWeightVector
        val rnd = new Random()

        partition.foreach(example => {
          val vector = example.only
          val scores = model.scoreFlatFeature(vector)
          val labels = vector.get(options.labelFamily)
          val labelMap = labels.iterator.map(fv => (fv.feature.name, fv.value)).toMap

          if (labels != null && labels.size() > 0) {
            val posLabels = labelMap.toArray
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
              val (idx, label, margin, iter) = pickRandomOtherLabel(model, labelMap, posIdx, posMargin, rnd, dim)
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
                val fvProjection = model.projectFeatureToEmbedding(vector)
                // update w-
                val negLabelKey = options.registry.feature(LABEL_EMBEDDING_KEY, negLabel)
                val gradContainerNeg = gradient.getOrElse(negLabelKey,
                  GradientContainer(new FloatVector(options.embeddingDimension), 0.0))
                gradContainerNeg.grad.multiplyAdd(rankLoss, fvProjection)
                // the real square sum can be computed by:
                // featureSquaredSum += Math.max(fvProjection.dot(fvProjection), 1.0)
                // but with rprop solver, this is not used, so we don't compute it here to improve speed
                gradient.put(negLabelKey, GradientContainer(gradContainerNeg.grad, 0.0))
                // update w+
                val posLabelKey = options.registry.feature(LABEL_EMBEDDING_KEY, posLabel)
                val gradContainerPos = gradient.getOrElse(posLabelKey,
                  GradientContainer(new FloatVector(options.embeddingDimension), 0.0))
                gradContainerPos.grad.multiplyAdd(-rankLoss, fvProjection)
                gradient.put(posLabelKey, GradientContainer(gradContainerPos.grad, 0.0))

                // compute gradient w.r.t V (featureWeightVector)
                val posLabelWeightVector = labelWeightVector.get(posLabel)
                val negLabelWeightVector = labelWeightVector.get(negLabel)
                for (fv <- vector.iterator) {
                  // We only care about features in the model.
                  if (featureWeightVector.containsKey(fv.feature)) {
                    val featureVal = fv.value
                    val gradContainer = gradient.getOrElse(fv.feature,
                      GradientContainer(new FloatVector(options.embeddingDimension), 0.0))
                    gradContainer.grad.multiplyAdd(featureVal.toFloat * rankLoss, negLabelWeightVector)
                    gradContainer.grad.multiplyAdd(-featureVal.toFloat * rankLoss, posLabelWeightVector)
                    gradient.put(fv.feature, GradientContainer(gradContainer.grad, 0.0))
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
                         key : String,
                         registry: FeatureRegistry) = {
    val model = train(sc, input, config, key, registry)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }

  private def normalizeWeightVectors(model: LowRankLinearModel, maxNorm: Double) = {
    for (weight <- model.featureWeightVector.values()) {
      weight.capNorm(maxNorm.toFloat)
    }

    for (weight <- model.labelWeightVector.values) {
      weight.capNorm(maxNorm.toFloat)
    }
  }

  private def parseTrainingOptions(config : Config, registry: FeatureRegistry)
    : LowRankLinearTrainerOptions = {
    LowRankLinearTrainerOptions(
      loss = config.getString("loss"),
      iterations = config.getInt("iterations"),
      labelFamily = registry.family(config.getString("rank_key")),
      lambda = config.getDouble("lambda"),
      subsample = config.getDouble("subsample"),
      minCount = config.getInt("min_count"),
      cache = Try(config.getString("cache")).getOrElse(""),
      solver = Try(config.getString("solver")).getOrElse("rprop"),
      embeddingDimension = config.getInt("embedding_dimension"),
      rankLossType = Try(config.getString("rank_loss")).getOrElse("non_uniform"),
      maxNorm = Try(config.getDouble("max_norm")).getOrElse(1.0),
      registry = registry
    )
  }

  private def pickRandomOtherLabel(model: LowRankLinearModel,
                                   posLabels: Map[String, Double],
                                   posIdx: Int,
                                   posMargin: Double,
                                   rnd: Random,
                                   dim: Int) : (Int, String, Double, Int) = {
    // Pick a random other label. This can be a negative or a positive with a smaller margin.
    var negIdx = rnd.nextInt(dim)
    var negLabel = model.labelDictionary.get(negIdx).getLabel
    var negMargin : Double = posLabels.getOrElse(negLabel, 0.0d)
    var iter = 0
    while ((negIdx == posIdx || negMargin > posMargin) && iter < dim) {
      // we only want to pick a label that has smaller margin, we try at most dim times
      // we try this for at most dim times
      negIdx = rnd.nextInt(dim)
      negLabel = model.labelDictionary.get(negIdx).getLabel
      negMargin = posLabels.getOrElse(negLabel, 0.0d)
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
    val model = new LowRankLinearModel(options.registry)
    val featureWeights = model.featureWeightVector
    val labelWeights = model.labelWeightVector
    val dict = model.labelDictionary
    val embeddingSize = options.embeddingDimension
    model.embeddingDimension(embeddingSize)
    var count : Int = 0
    for ((feature, featureStats) <- stats) {
      if (feature.family == options.labelFamily) {
        val entry = new LabelDictionaryEntry()
        entry.setLabel(feature.name)
        entry.setCount(featureStats.count.toInt)
        dict.add(entry)
      } else {
        if (!featureWeights.containsKey(feature)) {
          count = count + 1
          featureWeights.put(feature, FloatVector.getUniformVector(embeddingSize))
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
    log.info(s"Total number of inputFeatures is $count")

    model
  }
}
