package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core.features.{Feature, FeatureRegistry, Family}
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
                                          labelFamily : Family,
                                          lambda : Double,
                                          subsample : Double,
                                          minCount : Int,
                                          cache : String,
                                          solver : String,
                                          labelMinCount: Option[Int],
                                          registry: FeatureRegistry)

  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String,
            registry: FeatureRegistry) : FullRankLinearModel = {
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
                         key : String,
                         registry: FeatureRegistry) = {
    val model = train(sc, input, config, key, registry)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }
  
  def modelIteration(sc : SparkContext,
                     options : FullRankLinearTrainerOptions,
                     model : FullRankLinearModel,
                     pointwise : RDD[Example]) = {
    var prevGradients : Map[Feature, GradientContainer] = Map()
    val step = scala.collection.mutable.HashMap[Feature, FloatVector]()
    for (iter <- 0 until options.iterations) {
      log.info(s"Iteration $iter")
      val sample = pointwise.sample(false, options.subsample)
      val gradients: Map[Feature, GradientContainer] = options.loss match {
        case "softmax" => softmaxGradient(sc, options, model, sample)
        case "hinge" => hingeGradient(sc, options ,model, sample, "l1")
        case "squared_hinge" => hingeGradient(sc, options, model, sample, "l2")
        case _: String => softmaxGradient(sc, options, model, sample)
      }
      val weightVector = model.weightVector
      val dim = model.labelDictionary.size
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
    val toDelete = scala.collection.mutable.ArrayBuffer[Feature]()
    for (entry <- model.weightVector) {
      if (entry._2.dot(entry._2) < 1e-6) {
        toDelete.add(entry._1)
      }
    }
    // TODO (Brad): Encapsulation
    for (deleteFeature <- toDelete) {
      model.weightVector.remove(deleteFeature)
    }
  }
  
  def softmaxGradient(sc : SparkContext,
                      options : FullRankLinearTrainerOptions,
                      model : FullRankLinearModel,
                      pointwise : RDD[Example]) : Map[Feature, GradientContainer] = {
    val modelBC = sc.broadcast(model)
    
    pointwise
    .mapPartitions(partition => {
      val model = modelBC.value
      val labelToIdx = model.labelToIndex
      val dim = model.labelDictionary.size
      val gradient = scala.collection.mutable.HashMap[Feature, GradientContainer]()
      val weightVector = model.weightVector()

      partition.foreach(example => {
        val vector = example.only
        val labels = vector.get(options.labelFamily)
        if (labels != null) {
          val posLabels = labels.iterator.map(fv => fv.feature.name)
          val scores = model.scoreFlatFeature(vector)
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
          for (fv <- vector.iterator) {
            val key = fv.feature
            // We only care about features in the model.
            if (weightVector.containsKey(key)) {
              val gradContainer = gradient.getOrElse(key,
                                                     GradientContainer(new FloatVector(dim), 0.0))
              gradContainer.grad.multiplyAdd(fv.value, scores)
              val norm = math.max(fv.value * fv.value, 1.0)
              gradient.put(key,
                           GradientContainer(gradContainer.grad,
                           gradContainer.featureSquaredSum + norm
              ))
            }
          }
        }
      })
      gradient.iterator
    })
    .reduceByKey((a, b) => GradientUtils.sumGradients(a,b))
    .collectAsMap()
    .toMap
  }

  def hingeGradient(sc : SparkContext,
                    options : FullRankLinearTrainerOptions,
                    model : FullRankLinearModel,
                    pointwise : RDD[Example],
                    lossType : String) : Map[Feature, GradientContainer] = {
    val modelBC = sc.broadcast(model)

    pointwise
      .mapPartitions(partition => {
        val model = modelBC.value
        val labelToIdx = model.labelToIndex
        val dim = model.labelDictionary.size
        val gradient = scala.collection.mutable.HashMap[Feature, GradientContainer]()
        val weightVector = model.weightVector
        val rnd = new Random()

        partition.foreach(examples => {
          val vector = examples.only
          val labels = vector.get(options.labelFamily)
          if (labels != null && labels.size() > 0) {
            val posLabels = labels.iterator.map(fv => (fv.feature.name, fv.value)).toArray
            // Pick a random positive label
            val posLabelRnd = rnd.nextInt(posLabels.length)
            val (posLabel, posMargin) = posLabels(posLabelRnd)
            val posIdx = labelToIdx.get(posLabel)
            // Pick a random other label. This can be a negative or a positive with a smaller margin.
            var negIdx = rnd.nextInt(dim)
            while (negIdx == posIdx) {
              negIdx = rnd.nextInt(dim)
            }
            val negLabel = model.labelDictionary.get(negIdx).getLabel
            val negLabelFeature = options.labelFamily.feature(negLabel)
            val negMargin : Double = if (labels.containsKey(negLabelFeature)) {
              labels.getDouble(negLabelFeature)
            } else 0.0

            if (posMargin > negMargin) {
              val scores = model.scoreFlatFeature(vector)
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

                for (fv <- vector.iterator) {
                  val key = fv.feature
                  // We only care about features in the model.
                  if (weightVector.containsKey(key)) {
                    val featureVal = fv.value
                    val gradContainer = gradient.getOrElse(key,
                                                           GradientContainer(new FloatVector(dim), 0.0))
                    gradContainer.grad.multiplyAdd(featureVal, grad)
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
        })
        gradient.iterator
      })
      .reduceByKey((a, b) => GradientUtils.sumGradients(a,b))
      .collectAsMap
      .toMap
  }

  def parseTrainingOptions(config : Config, registry: FeatureRegistry) : FullRankLinearTrainerOptions = {
    FullRankLinearTrainerOptions(
        loss = config.getString("loss"),
        iterations = config.getInt("iterations"),
        labelFamily = registry.family(config.getString("rank_key")),
        lambda = config.getDouble("lambda"),
        subsample = config.getDouble("subsample"),
        minCount = config.getInt("min_count"),
        cache = Try(config.getString("cache")).getOrElse(""),
        solver = Try(config.getString("solver")).getOrElse("rprop"),
        labelMinCount = Try(Some(config.getInt("label_min_count"))).getOrElse(None),
        registry = registry
    )
  }
  
  def setupModel(options : FullRankLinearTrainerOptions, pointwise : RDD[Example]) : FullRankLinearModel = {
    val stats = TrainingUtils.getFeatureStatistics(options.minCount, pointwise)
    val labelCounts = if (options.labelMinCount.isDefined) {
      TrainingUtils.getLabelCounts(options.labelMinCount.get, pointwise, options.labelFamily)
    } else {
      TrainingUtils.getLabelCounts(options.minCount, pointwise, options.labelFamily)
    }

    val model = new FullRankLinearModel(options.registry)
    val weights = model.weightVector
    val dict = model.labelDictionary

    for (kv <- stats) {
      val (feature, _) = kv
      if (feature.family != options.labelFamily) {
        if (!weights.containsKey(feature)) {
          // Dummy entry until we know the number of labels.
          // TODO (Brad): Awkward. Intentionally setting the key to null in a map is scary.
          weights.put(feature, null)
        }
      }
    }

    for (kv <- labelCounts) {
      val (feature, count) = kv
      val entry = new LabelDictionaryEntry()
      entry.setLabel(feature.name())
      entry.setCount(count)
      dict.add(entry)
    }

    val dim = dict.size()
    log.info(s"Total number of labels is $dim")

    // Now fill all the feature vectors with length dim.
    var count : Int = 0
    for (feature <- weights.keySet()) {
      count = count + 1
      weights.put(feature, new FloatVector(dim))
    }
    log.info(s"Total number of inputFeatures is $count")
    model.buildLabelToIndex()

    model
  }
}
