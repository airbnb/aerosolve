package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.core.features.{Family, Feature, FeatureRegistry}
import com.typesafe.config.Config
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConversions._

object FeatureSelection {
  private final val log: Logger = LoggerFactory.getLogger("FeatureSelection")
  val allKey : (String, String) = ("$ALL", "$POS")

  // Given a RDD compute the pointwise mutual information between
  // the positive label and the discrete features.
  def pointwiseMutualInformation(examples : RDD[Example],
                                 config : Config,
                                 key : String,
                                 labelFamily : Family,
                                 posThreshold : Double,
                                 minPosCount : Double,
                                 newCrosses : Boolean,
                                  registry : FeatureRegistry) : RDD[(Feature, Double)] = {
    val pointwise = LinearRankerUtils.makePointwiseFloat(examples, config, key, registry)
    val allFeature = registry.feature(allKey._1, allKey._2)

    val features = pointwise
      .mapPartitions(part => {
        // The tuple2 is var, var | positive
        val output = scala.collection.mutable.HashMap[Feature, (Double, Double)]()
        part.foreach(example =>{
          val featureVector = example.only
          val labelVal = featureVector.get(labelFamily).iterator.next.value
          val isPos = if (labelVal > posThreshold) 1.0 else 0.0
          val all : (Double, Double) = output.getOrElse(allFeature, (0.0, 0.0))
          output.put(allFeature, (all._1 + 1.0, all._2 + 1.0 * isPos))

          if (newCrosses) {
            for (fv1 <- featureVector.iterator) {
              for (fv2 <- featureVector.iterator) {
                if (fv1.feature.compareTo(fv2.feature) <= 0) {
                  val feature = registry.feature(
                    "%s<NEW>%s".format(fv1.feature.family.name, fv2.feature.family.name),
                     "%s<NEW>%s".format(fv1.feature.name, fv2.feature.name))
                  val x = output.getOrElse(feature, (0.0, 0.0))
                  output.put(feature, (x._1 + 1.0, x._2 + 1.0 * isPos))
                }
              }
            }
          }
          for (featureValue <- featureVector.iterator()) {
            val x = output.getOrElse(featureValue.feature, (0.0, 0.0))
            output.put(featureValue.feature, (x._1 + 1.0, x._2 + 1.0 * isPos))
          }
        })
        output.iterator
      })
      .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
      .filter(x => x._2._2 >= minPosCount)

    val allCount = features.filter(x => x._1.equals(allFeature)).take(1).head

    features.map(x => {
      val prob = x._2._1 / allCount._2._1
      val probPos = x._2._2 / allCount._2._2
      (x._1, math.log(probPos / prob) / math.log(2.0))
    })
  }

  // Returns the maximum entropy per family
  def maxEntropy(input : RDD[((String, String), Double)]) : RDD[((String, String), Double)] = {
    input
      .map(x => (x._1._1, (x._1._2, x._2)))
      .reduceByKey((a, b) => if (math.abs(a._2) > math.abs(b._2)) a else b)
      .map(x => ((x._1, x._2._1), x._2._2))
  }
}
