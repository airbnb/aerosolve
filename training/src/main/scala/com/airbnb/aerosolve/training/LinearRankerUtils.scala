package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.core.FeatureVector
import com.airbnb.aerosolve.core.features._
import com.airbnb.aerosolve.core.models.AbstractModel
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

case class CompressedExample(pos : Array[Feature],
                             neg : Array[Feature],
                             label : Double)

object LinearRankerUtils {
  private final val log: Logger = LoggerFactory.getLogger("LinearRankerUtils")

  def getLabel(vector : MultiFamilyVector, labelFamily: Family): Int = {
    vector.get(labelFamily).iterator.next.value.toInt
  }

  def getFeatures(vector : Iterable[FeatureValue]) : Iterable[Feature] = {
    vector.map(fv => fv.feature)
  }

  // Does feature expansion on an example and buckets them by rank.
  // Assumes the example is transformed and contains a label.
  def expandAndBucketizeExample(example : Iterable[MultiFamilyVector],
                                labelFamily : Family) :
  Array[Array[Iterable[Feature]]] = {
    example
      .map(sample => {
        val labelBucket : Int = getLabel(sample, labelFamily)
        val features = getFeatures(sample)
        (labelBucket, features)
      })
      .groupBy(_._1)
      .toSeq
      // Sort buckets in ascending order.
      .sortBy(_._1)
      .map{ case (_, iter) => iter.map(_._2).toArray }
      .toArray
  }

  def rankingCompression(example : Iterable[MultiFamilyVector], labelFamily : Family) : Seq[CompressedExample] = {
    val output = ArrayBuffer[CompressedExample]()
    val buckets = expandAndBucketizeExample(example, labelFamily)
    val rnd = new Random()
    for (i <- 0 to buckets.length - 2) {
      for (j <- i + 1 to buckets.length - 1) {
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
    output.toSeq
  }

  def score(vector : Iterable[Feature],
            weightMap : collection.mutable.Map[Feature, (Double, Double)]) : Double = {
    var sum : Double = 0
    vector.iterator.foreach(feature => {
      val opt = weightMap.get(feature)
      if (opt.isDefined) {
        sum += opt.get._1
      }
    })
    sum
  }

  // Makes an example pointwise while preserving the float features.
  def makePointwiseFloat(
                          examples : RDD[Example],
                          config : Config,
                          key : String,
                          registry: FeatureRegistry) : RDD[Example] = {
    val transformer = new Transformer(config, key, registry)
    examples.flatMap(example => {
      example.map(vector => {
        // TODO (Brad): Why make a new one? Also, why make separate examples for each vector?
        val newExample = new SimpleExample(registry)
        newExample.context().merge(example.context())
        newExample.createVector().merge(vector)
        newExample.transform(transformer)
      })
    })
  }
  
  // Transforms an RDD of examples
  def transformExamples(
                        examples : RDD[Example],
                        config : Config,
                        key : String,
                        registry: FeatureRegistry) : RDD[Example] = {
    val transformer = new Transformer(config, key, registry)
    examples.map(_.transform(transformer))
  }

  // TODO (Brad): I removed makePointwise. It removed all float features besides the label family.
  // Now that we don't distinguish float and String, I'm not sure how to do this and I'm worried
  // it will break something.  The comment said it was for space reasons and if so,
  // maybe we can skip it with the more efficient representation.
}
