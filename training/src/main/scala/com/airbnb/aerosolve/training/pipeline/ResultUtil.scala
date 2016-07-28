package com.airbnb.aerosolve.training.pipeline

import com.airbnb.aerosolve.training._
import scala.collection.mutable

/*
 * Writes to result file
 *
 * Format:
    {
      "HOLD_PRECISION_RECALL_AUC": 0.897908,
      "HOLD_ACC": 0.897908,
      ...
      "HOLD_THRESHOLDS": [
        0.021060,
        0.313343,
        0.897908,
        1.190191,
        1.482474
      ],
      "HOLD_RECALLS": [
        0.021060,
        0.313343,
        0.897908,
        1.190191,
        1.482474
      ],
      "HOLD_PRECISIONS": [
        0.021060,
        0.313343,
        0.897908,
        1.190191,
        1.482474
      ],
      "TRAIN_PRECISION_RECALL_AUC": 0.897908,
      "TRAIN_ACC": 0.897908,
      ...
      "TRAIN_THRESHOLDS": [
        0.021060,
        0.313343,
        0.897908,
        1.190191,
        1.482474
      ],
      "TRAIN_RECALLS": [
        0.021060,
        0.313343,
        0.897908,
        1.190191,
        1.482474
      ],
      "TRAIN_PRECISIONS": [
        0.021060,
        0.313343,
        0.897908,
        1.190191,
        1.482474
      ]
    }
 *
 */
object ResultUtil {
  def writeResults(resultsOutputPath: String,
                   metrics: Array[(String, Double)],
                   holdThresholdPrecisionRecall: Array[(Double, Double, Double)],
                   trainThresholdPrecisionRecall: Array[(Double, Double, Double)]) = {
    var allMetrics = mutable.Buffer[String]()

    // Add main metrics
    for (i <- 0 until metrics.length - 1) {
      var name = metrics(i)._1
      if (name(0) == '!') {
        name = name.substring(1)
      }
      val value = metrics(i)._2
      allMetrics.append("\"" + name + "\": " + value)
    }

    // Add hold precision recall metrics
    var holdThresholds = mutable.Buffer[String]()
    var holdPrecisions = mutable.Buffer[String]()
    var holdRecalls = mutable.Buffer[String]()
    for (i <- 0 until holdThresholdPrecisionRecall.length - 1) {
      val elem = holdThresholdPrecisionRecall(i)
      holdThresholds.append(elem._1.toString)
      holdPrecisions.append(elem._2.toString)
      holdRecalls.append(elem._3.toString)
    }
    allMetrics.append(holdThresholds.mkString("\"HOLD_THRESHOLDS\": [", ",", "]"))
    allMetrics.append(holdPrecisions.mkString("\"HOLD_PRECISIONS\": [", ",", "]"))
    allMetrics.append(holdRecalls.mkString("\"HOLD_RECALLS\": [", ",", "]"))

    // Add hold precision recall metrics
    var trainThresholds = mutable.Buffer[String]()
    var trainPrecisions = mutable.Buffer[String]()
    var trainRecalls = mutable.Buffer[String]()
    for (i <- 0 until trainThresholdPrecisionRecall.length - 1) {
      val elem = trainThresholdPrecisionRecall(i)
      trainThresholds.append(elem._1.toString)
      trainPrecisions.append(elem._2.toString)
      trainRecalls.append(elem._3.toString)
    }
    allMetrics.append(trainThresholds.mkString("\"TRAIN_THRESHOLDS\": [", ",", "]"))
    allMetrics.append(trainPrecisions.mkString("\"TRAIN_PRECISIONS\": [", ",", "]"))
    allMetrics.append(trainRecalls.mkString("\"TRAIN_RECALLS\": [", ",", "]"))

    // Write to file as a string
    val json = allMetrics.mkString("{", ",", "}")
    PipelineUtil.writeStringToFile(json, resultsOutputPath)
  }
}
