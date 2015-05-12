package com.airbnb.aerosolve.training

import java.util.concurrent.ConcurrentHashMap
import java.io.{BufferedWriter, OutputStreamWriter}

import com.airbnb.aerosolve.training.CyclicCoordinateDescent.Params
import com.airbnb.aerosolve.core.models.SplineModel
import com.airbnb.aerosolve.core.models.SplineModel.WeightSpline
import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.core.{Example, FeatureVector}
import com.typesafe.config.Config
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

object SplineQualityMetrics {
  private final val log: Logger = LoggerFactory.getLogger("SplineQualityMetrics")

  // This object calculates a number of metrics (smoothness, monotonicity) of splines
  // that can be used to evaluate the quality and explainability of splines when we
  // generate new versions of Appraiser.

  // Calculate lag-one autocorrelation as a proxy for smoothness.
  // 1 = smooth, 0 = non-smooth, -1 means noisy around average (higher than average is usually followed by
  // lower than average)
  def getSmoothness(data : Array[Double]) : (Double) = {
    val a = data.slice(0, data.length - 1)
    val b = data.slice(1, data.length)
    return getCorrelation(a, b)
  }

  def getCorrelation(dataOne: Array[Double], dataTwo: Array[Double]) : (Double) = {
    var totalOne = 0.0
    var totalTwo = 0.0
    for (value <- dataOne) {
      totalOne += value
    }
    for (value <- dataTwo) {
      totalTwo += value
    }
    val meanOne = totalOne / dataOne.length
    val meanTwo = totalTwo / dataTwo.length
    val normOne = subRowsArray(dataOne, Array.fill[Double](dataOne.length)(meanOne), dataOne.length)
    val normTwo = subRowsArray(dataTwo, Array.fill[Double](dataTwo.length)(meanTwo), dataTwo.length)
    var rNum = 0.0
    var sumSquaresOne = 0.0
    var sumSquaresTwo = 0.0
    for (i <- 0 until normOne.length) {
      rNum += normOne(i) * normTwo(i)
      sumSquaresOne += Math.pow(normOne(i), 2.0)
      sumSquaresTwo += Math.pow(normTwo(i), 2.0)
    }
    val rDen = Math.sqrt(sumSquaresOne * sumSquaresTwo)
    var r = rNum / rDen
    r = Math.max(Math.min(r, 1.0), -1.0)
    r
  }

  private def subRowsArray(a: Array[Double], b: Array[Double], sizeHint: Int): Array[Double] = {
    val l: Array[Double] = new Array(sizeHint)
    var i = 0
    while (i < sizeHint) {
      l(i) = a(i) - b(i)
      i += 1
    }
    l
  }

  // Uses the distance in damerauLevenshteinDistance as a proxy for monotonicity.
  def getMonotonicity(data : Array[Double]) : (Double) = {
    var monotonicityUnNorm = math.min(damerauLevenshteinDistance(data, data.sorted),
      damerauLevenshteinDistance(data, data.sorted.reverse))
    val monotonicity = 1.0 - (monotonicityUnNorm / data.length.toDouble)
    return monotonicity
  }

  // Calculates the distance according to this algorithm:
  // http://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance#Optimal_string_alignment_distance
  private def damerauLevenshteinDistance(dataOne : Array[Double],
                                 dataTwo : Array[Double]) : (Double) = {
    val rows = dataOne.length + 1
    val cols = dataTwo.length + 1
    var dlMatrix = Array.ofDim[Double](rows, cols)
    var cost = 0
    for( i <- 0 to dataOne.length){
      dlMatrix(i)(0) = 1.0*i
    }
    for( j <- 1 to dataTwo.length){
      dlMatrix(0)(j) = 1.0*j
    }
    for( i <- 0 until dataOne.length){
      for( j <- 0 until dataTwo.length){
        if (dataOne(i) == dataTwo(i)){
          cost = 0
        } else {
          cost = 1
        }
        dlMatrix(i+1)(j+1) = math.min(math.min(dlMatrix(i)(j+1) + 1, dlMatrix(i+1)(j) + 1), dlMatrix(i)(j) + cost)
        if (i > 0 && j > 0 && dataOne(i) == dataTwo(j-1) && dataOne(i-1) == dataTwo(j)){
          dlMatrix(i+1)(j+1) = math.min(dlMatrix(i+1)(j+1), dlMatrix(i-1)(j-1) + cost)
        }
      }
    }
    return dlMatrix(dataOne.length)(dataTwo.length)
  }

}