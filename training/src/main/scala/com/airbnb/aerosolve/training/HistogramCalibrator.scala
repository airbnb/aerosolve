package com.airbnb.aerosolve.training

import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ArrayBuffer

// Calibrates scores into the [0 .. 1] range by taking a histogram of scores
// creating a cumulative distribution function and then finding
// a mapping from the score to the y-axis of the cumulative distribution function.
class HistogramCalibrator extends Serializable {
  // Mapping of score to cumulative distribution.
  var cumulativeDistributionFunction : Array[(Double, Double)] = Array()

  def setFromDoubleRDD(scores : RDD[Double], numBuckets : Int) = {
    val histogram = scores.histogram(numBuckets)
    // Spark returns one more bucket than the size of counts.
    val buckets = histogram._1
    val count = histogram._2
    val cum = ArrayBuffer[(Double, Double)]()
    cum += Tuple2(buckets.head, 0.0)
    for (i <- 1 until buckets.size) {
      val next = Tuple2(buckets(i), cum.last._2 + count(i - 1))
      cum += next
    }
    val scale = if (cum.last._2 > 0) {
      1.0 / cum.last._2
    } else {
      1.0
    }
    cumulativeDistributionFunction = cum
      .map(x => (x._1, x._2 * scale))
      .toArray
  }

  // Returns the element index whose value equals to or just lower than score.
  def lowerBound(score : Double, start : Int, end : Int) : Int = {
    if (end - start <= 1) {
      return start
    }
    val mid = (start + end) / 2
    if (score < cumulativeDistributionFunction(mid)._1) {
      lowerBound(score, start, mid)
    } else {
      lowerBound(score, mid, end)
    }
  }

  def calibrateScore(score : Double) : Double = {
    if (score <= cumulativeDistributionFunction.head._1) {
      return 0.0
    }
    if (score >= cumulativeDistributionFunction.last._1) {
      return 1.0
    }
    val lower = lowerBound(score, 0, cumulativeDistributionFunction.length)
    assert(lower < cumulativeDistributionFunction.length)
    val lowerVal = cumulativeDistributionFunction(lower)
    val upperVal = cumulativeDistributionFunction(lower + 1)
    // Linearly interpolate between the buckets
    val u = (score - lowerVal._1) /
            (upperVal._1 - lowerVal._1)
    if (u <= 0.0) {
      return lowerVal._2
    } else if (u >= 1.0) {
      return upperVal._2
    }
    return upperVal._2 * u + (1.0 - u) * lowerVal._2
  }
}

// Companion object
object HistogramCalibrator {
  def apply(scores : RDD[Double], buckets : Int) = {
    val calibrator = new HistogramCalibrator()
    calibrator.setFromDoubleRDD(scores, buckets)
    calibrator
  }
}