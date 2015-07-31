package com.airbnb.aerosolve.training

import java.io.BufferedWriter
import java.io.OutputStreamWriter
import java.util

import com.airbnb.aerosolve.core.{ModelRecord, ModelHeader, FeatureVector, Example}
import com.airbnb.aerosolve.core.models.LinearModel
import com.airbnb.aerosolve.core.util.Util
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
import scala.math.abs
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path

object FeatureSelection {
  private final val log: Logger = LoggerFactory.getLogger("FeatureSelection")
  val allKey : (String, String) = ("$ALL", "$POS")

  // Given a RDD compute the pointwise mutual information between
  // the positive label and the discrete features.
  def pointwiseMutualInformation(examples : RDD[Example],
                                 config : Config,
                                 key : String,
                                 rankKey : String,
                                 posThreshold : Double,
                                 minPosCount : Double,
                                 newCrosses : Boolean) : RDD[((String, String), Double)] = {
    val pointwise = LinearRankerUtils.makePointwise(examples, config, key, rankKey)
    val features = pointwise
      .mapPartitions(part => {
      // The tuple2 is var, var | positive
      val output = scala.collection.mutable.HashMap[(String, String), (Double, Double)]()
      part.foreach(example =>{
        val featureVector = example.example.get(0)
        val isPos = if (featureVector.floatFeatures.get(rankKey).asScala.head._2 > posThreshold) 1.0
        else 0.0
        val all : (Double, Double) = output.getOrElse(allKey, (0.0, 0.0))
        output.put(allKey, (all._1 + 1.0, all._2 + 1.0 * isPos))

        val features : Array[(String, String)] =
          LinearRankerUtils.getFeatures(featureVector)
        if (newCrosses) {
          for (i <- features) {
            for (j <- features) {
              if (i._1 < j._1) {
                val key = ("%s<NEW>%s".format(i._1, j._1),
                           "%s<NEW>%s".format(i._2, j._2))
                val x = output.getOrElse(key, (0.0, 0.0))
                output.put(key, (x._1 + 1.0, x._2 + 1.0 * isPos))
              }
            }
          }
        }
        for (feature <- features) {
          val x = output.getOrElse(feature, (0.0, 0.0))
          output.put(feature, (x._1 + 1.0, x._2 + 1.0 * isPos))
        }
      })
      output.iterator
    })
    .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
    .filter(x => x._2._2 >= minPosCount)

    val allCount = features.filter(x => x._1.equals(allKey)).take(1).head

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
