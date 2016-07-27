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
      "hold_threshold": [
        0.021060,
        0.313343,
        0.897908,
        1.190191,
        1.482474
      ],
      "hold_recall": [
        0.021060,
        0.313343,
        0.897908,
        1.190191,
        1.482474
      ],
      "hold_precision": [
        0.021060,
        0.313343,
        0.897908,
        1.190191,
        1.482474
      ],
      "TRAIN_PRECISION_RECALL_AUC": 0.897908,
      "TRAIN_ACC": 0.897908,
      ...
      "train_threshold": [
        0.021060,
        0.313343,
        0.897908,
        1.190191,
        1.482474
      ],
      "train_recall": [
        0.021060,
        0.313343,
        0.897908,
        1.190191,
        1.482474
      ],
      "train_precision": [
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
      val elem = metrics(i)
      allMetrics.append("\"" + elem._1 + "\": " + elem._2)
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
    allMetrics.append(holdThresholds.mkString("\"hold_threshold\": [", ",", "]"))
    allMetrics.append(holdPrecisions.mkString("\"hold_precision\": [", ",", "]"))
    allMetrics.append(holdRecalls.mkString("\"hold_recall\": [", ",", "]"))

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
    allMetrics.append(trainThresholds.mkString("\"train_threshold\": [", ",", "]"))
    allMetrics.append(trainPrecisions.mkString("\"train_precision\": [", ",", "]"))
    allMetrics.append(trainRecalls.mkString("\"train_recall\": [", ",", "]"))

    // Write to file as a string
    val json = allMetrics.mkString("{", ",", "}")
    PipelineUtil.writeStringToFile(json, resultsOutputPath)
  }
}
