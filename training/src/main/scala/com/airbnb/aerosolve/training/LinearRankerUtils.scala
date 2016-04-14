package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.core.FeatureVector
import com.airbnb.aerosolve.core.transforms.Transformer
import com.typesafe.config.Config
import org.slf4j.{LoggerFactory, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Buffer
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.util.Random

case class CompressedExample(pos : Array[(String, String)],
                             neg : Array[(String, String)],
                             label : Double);

object LinearRankerUtils {
  private final val log: Logger = LoggerFactory.getLogger("LinearRankerUtils")

  def getFeatures(sample : FeatureVector) : Array[(String, String)] = {
    val features = HashSet[(String, String)]()
    sample.getStringFeatures.foreach(family => {
      family._2.foreach(value => {
        features.add((family._1, value))
      })
    })
    features.toArray
  }

  // Does feature expansion on an example and buckets them by rank.
  def expandAndBucketizeExamples(
                                 examples : Example,
                                 transformer : Transformer,
                                 rankKey : String) :
  Array[Array[Array[(String, String)]]] = {
    transformer.combineContextAndItems(examples)
    val samples : Seq[FeatureVector] = examples.example
    val buckets = HashMap[Int, Buffer[Array[(String, String)]]]()
    samples
      .filter(x => x.getStringFeatures != null &&
                   x.getFloatFeatures != null &&
                   x.getFloatFeatures.get(rankKey) != null)
      .foreach(sample => {
      val rankBucket : Int = sample.getFloatFeatures.get(rankKey).toSeq.head._2.toInt
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
      .sortWith((x,y) => x._1 < y._1)
      .map(x => x._2.toArray)
      .toArray
  }

  // Makes ranking training data
  def rankingTrain(input : RDD[Example], config : Config, key : String) :
  RDD[CompressedExample] = {
    input
      .mapPartitions(partition => {
      val output = ArrayBuffer[CompressedExample]()
      val rnd = new Random()
      val rankKey: String = config.getString(key + ".rank_key")
      val transformer = new Transformer(config, key)
      partition.foreach(examples => {
        val buckets = LinearRankerUtils.expandAndBucketizeExamples(examples, transformer, rankKey)
        for (i <- 0 to buckets.size - 2) {
          for (j <- i + 1 to buckets.size - 1) {
            val neg = buckets(i)(rnd.nextInt(buckets(i).size)).toSet
            val pos = buckets(j)(rnd.nextInt(buckets(j).size)).toSet
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

  def score(feature : Array[(String, String)],
             weightMap : collection.mutable.Map[(String, String), (Double, Double)]) : Double = {
    var sum : Double = 0
    feature.foreach(v => {
      val opt = weightMap.get(v)
      if (opt != None) {
        sum += opt.get._1
      }
    })
    sum
  }

  // Makes an example pointwise while preserving the float features.
  def makePointwiseFloat(
                    examples : RDD[Example],
                    config : Config,
                    key : String) : RDD[Example] = {
    val transformer = new Transformer(config, key)
    examples.map(example => {
      val buffer = collection.mutable.ArrayBuffer[Example]()
      example.example.asScala.foreach(x => {
        val newExample = new Example()
        newExample.setContext(example.context)
        newExample.addToExample(x)
        transformer.combineContextAndItems(newExample)
        buffer.append(newExample)
      })
      buffer
    })
    .flatMap(x => x)
  }
  
  // Transforms an RDD of examples
  def transformExamples(
                    examples : RDD[Example],
                    config : Config,
                    key : String) : RDD[Example] = {
    val transformer = new Transformer(config, key)
    examples.map(example => {
        transformer.combineContextAndItems(example)
        example.unsetContext()
        example
      })
  }

  // Since examples are bags of user impressions, for pointwise algoriths
  // we need to shuffle each feature vector separately.
  def makePointwise(examples : RDD[Example],
                    config : Config,
                    key : String,
                    rankKey : String) : RDD[Example] = {
    val transformer = new Transformer(config, key)
    examples.map(example => {
      val buffer = collection.mutable.ArrayBuffer[Example]()
      example.example.asScala.foreach{x => {
        val newExample = new Example()
        newExample.setContext(example.context)
        newExample.addToExample(x)
        transformer.combineContextAndItems(newExample)
        // For space reasons remove all float features except rankKey.
        val floatFeatures = newExample.example.get(0).getFloatFeatures
        if (floatFeatures != null) {
          val rank : java.util.Map[java.lang.String, java.lang.Double] =
            floatFeatures.get(rankKey)
          val newFloat : java.util.Map[java.lang.String,
            java.util.Map[java.lang.String, java.lang.Double]] =
            new java.util.HashMap()
          newFloat.put(rankKey, rank)
          newExample.example.get(0).setFloatFeatures(newFloat)
        }
        buffer.append(newExample)
      }}
      buffer
    })
      .flatMap(x => x)
  }

  def makePointwiseCompressed(examples : RDD[Example],
                    config : Config,
                    key : String) : RDD[CompressedExample] = {
    val rankKey: String = config.getString(key + ".rank_key")
    val pointwise = makePointwise(examples, config, key, rankKey)
    pointwise.map(example => {
      val ex = example.example.get(0)
      val label = ex.floatFeatures.get(rankKey).entrySet().iterator().next().getValue
      CompressedExample(getFeatures(ex), Array[(String, String)](), label)
    })
  }
}
