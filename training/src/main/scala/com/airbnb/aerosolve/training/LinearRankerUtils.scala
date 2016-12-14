package com.airbnb.aerosolve.training

import java.util

import com.airbnb.aerosolve.core.features.SparseLabeledPoint
import com.airbnb.aerosolve.core.models.AdditiveModel
import com.airbnb.aerosolve.core.transforms.Transformer
import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.core.{Example, FeatureVector}
import com.typesafe.config.Config
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

case class CompressedExample(pos: Array[(String, String)],
                             neg: Array[(String, String)],
                             label: Double)

object LinearRankerUtils {

  def getFeatures(sample: FeatureVector): Array[(String, String)] = {
    val features = mutable.HashSet[(String, String)]()
    sample.stringFeatures.foreach(family => {
      family._2.foreach(value => {
        features.add((family._1, value))
      })
    })
    features.toArray
  }

  def getNumFeatures(ex: util.Map[java.lang.String, util.Map[java.lang.String, java.lang.Double]],
                     rankKey: String): Int = {
    // Get the number of features in the example excluding the label
    var numFeature = 0

    for (family <- ex) {
      if (!family._1.equals(rankKey)) {
        numFeature += family._2.size
      }
    }
    numFeature
  }

  // Does feature expansion on an example and buckets them by rank.
  def expandAndBucketizeExamples(
                                  examples: Example,
                                  transformer: Transformer,
                                  rankKey: String):
  Array[Array[Array[(String, String)]]] = {
    transformer.combineContextAndItems(examples)
    val samples: Seq[FeatureVector] = examples.example
    val buckets = mutable.HashMap[Int, mutable.Buffer[Array[(String, String)]]]()
    samples
      .filter(x => x.stringFeatures != null &&
        x.floatFeatures != null &&
        x.floatFeatures.get(rankKey) != null)
      .foreach(sample => {
        val rankBucket: Int = sample.floatFeatures.get(rankKey).toSeq.head._2.toInt
        val features = getFeatures(sample)
        val entryOpt = buckets.get(rankBucket)
        if (entryOpt.isEmpty) {
          buckets.put(rankBucket, ArrayBuffer(features))
        } else {
          entryOpt.get.append(features)
        }
      })
    // Sort buckets in ascending order.
    buckets
      .toBuffer
      .sortWith((x, y) => x._1 < y._1)
      .map(x => x._2.toArray)
      .toArray
  }

  // Makes ranking training data
  def rankingTrain(input: RDD[Example], config: Config, key: String):
  RDD[CompressedExample] = {
    input
      .mapPartitions(partition => {
        val output = ArrayBuffer[CompressedExample]()
        val rnd = new Random()
        val rankKey: String = config.getString(key + ".rank_key")
        val transformer = new Transformer(config, key)
        partition.foreach(examples => {
          val buckets = LinearRankerUtils.expandAndBucketizeExamples(examples, transformer, rankKey)
          for (i <- 0 to buckets.length - 2) {
            for (j <- i + 1 until buckets.length) {
              val neg = buckets(i)(rnd.nextInt(buckets(i).length)).toSet
              val pos = buckets(j)(rnd.nextInt(buckets(j).length)).toSet
              val intersect = pos.intersect(neg)
              // For ranking we have pairs of examples with label always 1.0.
              val out = CompressedExample(pos.diff(intersect).toArray,
                neg.diff(intersect).toArray,
                label = 1.0)
              output.append(out)
            }
          }
        })
        output.iterator
      })
  }

  def score(feature: Array[(String, String)],
            weightMap: collection.mutable.Map[(String, String), (Double, Double)]): Double = {
    var sum: Double = 0
    feature.foreach(v => {
      val opt = weightMap.get(v)
      if (opt.isDefined) {
        sum += opt.get._1
      }
    })
    sum
  }

  /**
    * Makes an example pointwise by combining context and flatmap with feature transforms.
    * Each observation is further flattened to [[SparseLabeledPoint]] format where features are indexed
    * by a specified [[AdditiveModel]].
    */
  def makePointwiseFloatVector(examples: RDD[Example],
                               transformer: Transformer,
                               params: AdditiveModelTrainer.AdditiveTrainerParams,
                               modelBC: Broadcast[AdditiveModel],
                               isTraining: Example => Boolean = _ => true,
                               groupSize: Int = 100): RDD[SparseLabeledPoint] = {
    val assemblerTimer = examples.sparkContext.accumulator(0L, "pointAssembler")

    makePointwiseFloat(examples, transformer, groupSize)
      .mapPartitions {
        examples =>
          val featureIndex = modelBC.value.getFeatureIndexer
          val denseFeatureIndexer = featureIndex.get(AdditiveModel.DENSE_FAMILY)

          // reuse buffer for each example to avoid GC
          val indices = new ArrayBuffer[Int]()
          val values = new ArrayBuffer[Float]()
          // for dense features, each feature is a List<Double> to be converted to float[]
          val denseIndices = new ArrayBuffer[Int]()
          val denseValues = new ArrayBuffer[Array[Float]]()

          examples.map {
            example =>
              val t0 = System.nanoTime()

              // clear buffer from last example
              indices.clear()
              values.clear()
              denseIndices.clear()
              denseValues.clear()

              val featureVector = example.example.get(0)

              Util.flattenFeatureAsStream(featureVector)
                .iterator
                .foreach {
                  featureFamily =>
                    val familyIndex = featureIndex.get(featureFamily.getKey)
                    if (familyIndex != null) {
                      featureFamily.getValue.iterator().foreach {
                        feature =>
                          val index = familyIndex.get(feature.getKey)
                          if (index != null) {
                            indices += index.intValue()
                            values += feature.getValue.floatValue()
                          }
                      }
                    }
                }

              val denseFeatures = featureVector.denseFeatures
              if (denseFeatureIndexer != null && denseFeatures != null) {
                denseFeatures.iterator.foreach {
                  case (featureName, featureValues) =>
                    val index = denseFeatureIndexer.get(featureName)
                    if (index != null) {
                      denseIndices += index.intValue()

                      val featureLength = featureValues.size()
                      val valueArray = new Array[Float](featureLength)
                      var i = 0
                      while (i < featureLength) {
                        valueArray(i) = featureValues.get(i).floatValue()
                        i += 1
                      }

                      denseValues += valueArray
                    }
                }
              }

              val label =
                if (params.loss.function == AdditiveModelTrainer.LossFunctions.REGRESSION) {
                  TrainingUtils.getLabel(featureVector, params.rankKey)
                } else {
                  TrainingUtils.getLabel(featureVector, params.rankKey, params.threshold)
                }

              assemblerTimer += (System.nanoTime() - t0)

              new SparseLabeledPoint(
                isTraining(example),
                label,
                indices.toArray, values.toArray,
                denseIndices.toArray, denseValues.toArray
              )
          }
      }
  }

  /**
    * Makes an example pointwise by combining context and flatmap with feature transforms.
    * Transform is applied in batch fashion so some transformers can be more
    * efficiently applied. (e.g. XGBoostTransform)
    */
  def makePointwiseFloat(
                          examples: RDD[Example],
                          transformer: Transformer,
                          groupSize: Int = 100): RDD[Example] = {
    val transformerTimer = examples.sparkContext.accumulator(0L, "transformer")
    examples.flatMap(example => {
      transformer.transformContext(example.context)
      example.example.iterator().map(x => {
        transformer.transformItem(x)

        val newExample = new Example()
        newExample.setContext(example.context)
        newExample.addToExample(x)
        newExample.setMetadata(example.metadata)
        transformer.addContextToItems(newExample)

        newExample
      })
    }).mapPartitions(exampleIterator => {
      exampleIterator.grouped(groupSize).flatMap {
        examples =>
          val t0 = System.nanoTime()

          val features = examples.iterator.flatMap(_.example).toIterable.asJava
          transformer.transformCombined(features)
          transformerTimer += (System.nanoTime() - t0)

          examples
      }
    })
  }

  // Makes an example pointwise while preserving the float features.
  def makePointwiseFloat(examples: RDD[Example],
                         config: Config,
                         key: String): RDD[Example] = {
    val transformer = new Transformer(config, key)
    makePointwiseFloat(examples, transformer)
  }

  // Transforms an RDD of examples
  def transformExamples(
                         examples: RDD[Example],
                         config: Config,
                         key: String): RDD[Example] = {
    val transformer = new Transformer(config, key)
    examples.map(example => {
      transformer.combineContextAndItems(example)
      example.unsetContext()
      example
    })
  }

  // Since examples are bags of user impressions, for pointwise algoriths
  // we need to shuffle each feature vector separately.
  def makePointwise(examples: RDD[Example],
                    config: Config,
                    key: String,
                    rankKey: String): RDD[Example] = {
    val transformer = new Transformer(config, key)
    examples.map(example => {
      val buffer = collection.mutable.ArrayBuffer[Example]()
      example.example.asScala.foreach { x => {
        val newExample = new Example()
        newExample.setContext(example.context)
        newExample.addToExample(x)
        newExample.setMetadata(example.metadata)
        transformer.combineContextAndItems(newExample)
        // For space reasons remove all float features except rankKey.
        val floatFeatures = newExample.example.get(0).getFloatFeatures
        if (floatFeatures != null) {
          val rank: java.util.Map[java.lang.String, java.lang.Double] =
            floatFeatures.get(rankKey)
          val newFloat: java.util.Map[java.lang.String,
            java.util.Map[java.lang.String, java.lang.Double]] =
            new java.util.HashMap()
          newFloat.put(rankKey, rank)
          newExample.example.get(0).setFloatFeatures(newFloat)
        }
        buffer.append(newExample)
      }
      }
      buffer
    })
      .flatMap(x => x)
  }

  def makePointwiseCompressed(examples: RDD[Example],
                              config: Config,
                              key: String): RDD[CompressedExample] = {
    val rankKey: String = config.getString(key + ".rank_key")
    val pointwise = makePointwise(examples, config, key, rankKey)
    pointwise.map(example => {
      val ex = example.example.get(0)
      val label = ex.floatFeatures.get(rankKey).entrySet().iterator().next().getValue
      CompressedExample(getFeatures(ex), Array[(String, String)](), label)
    })
  }
}
