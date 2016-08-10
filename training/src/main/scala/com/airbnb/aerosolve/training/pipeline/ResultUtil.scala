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
                   trainThresholdPrecisionRecall: Array[(Double, Double, Double)],
                   writeResult: Boolean = true) = {
    var allMetrics = mutable.Buffer[String]()

    // Add main metrics
    for (i <- 0 until metrics.length) {
      var name = metrics(i)._1
      if (name(0) == '!') {
        name = name.substring(1)
      }
      val value = metrics(i)._2
      if (!value.isNaN) {
        allMetrics.append("\"" + name + "\": " + value)
      }
    }

    // Add hold precision recall metrics
    var holdThresholds = mutable.Buffer[String]()
    var holdPrecisions = mutable.Buffer[String]()
    var holdRecalls = mutable.Buffer[String]()
    for (i <- 0 until holdThresholdPrecisionRecall.length - 1) {
      val elem = holdThresholdPrecisionRecall(i)
      holdThresholds.append("\t" + elem._1.toString)
      holdPrecisions.append("\t" + elem._2.toString)
      holdRecalls.append("\t" + elem._3.toString)
    }
    allMetrics.append(holdThresholds.mkString("\"HOLD_THRESHOLDS\": [\n", ",\n", "\n]"))
    allMetrics.append(holdPrecisions.mkString("\"HOLD_PRECISIONS\": [\n", ",\n", "\n]"))
    allMetrics.append(holdRecalls.mkString("\"HOLD_RECALLS\": [\n", ",\n", "\n]"))

    // Add train precision recall metrics
    var trainThresholds = mutable.Buffer[String]()
    var trainPrecisions = mutable.Buffer[String]()
    var trainRecalls = mutable.Buffer[String]()
    for (i <- 0 until trainThresholdPrecisionRecall.length - 1) {
      val elem = trainThresholdPrecisionRecall(i)
      trainThresholds.append("\t" + elem._1.toString)
      trainPrecisions.append("\t" + elem._2.toString)
      trainRecalls.append("\t" + elem._3.toString)
    }
    allMetrics.append(trainThresholds.mkString("\"TRAIN_THRESHOLDS\": [\n", ",\n", "\n]"))
    allMetrics.append(trainPrecisions.mkString("\"TRAIN_PRECISIONS\": [\n", ",\n", "\n]"))
    allMetrics.append(trainRecalls.mkString("\"TRAIN_RECALLS\": [\n", ",\n", "\n]"))

    // Write to file as a string
    val jsonString = allMetrics.mkString("{\n", ",\n", "\n}\n")
    if (writeResult) {
      PipelineUtil.writeStringToFile(jsonString, resultsOutputPath)
    }

    // Return jsonString for testing
    jsonString
  }
}
