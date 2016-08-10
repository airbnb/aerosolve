package com.airbnb.aerosolve.training.pipeline

import org.junit.Assert._
import org.junit.Test

import scala.util.parsing.json.JSON

class ResultUtilTest {
  @Test
  def testExampleToEvaluationRecordMulticlass() = {
    val metrics = Array(("!HOLD_AUC", 0.11), ("TRAIN_ACC", 0.22))
    val holdMetrics = Array((0.1, 0.2, 0.3), (1.1, 1.2, 1.3))
    val trainMetrics = Array[(Double, Double, Double)]()
    val resultString = ResultUtil.writeResults("resultsOutputPath", metrics, holdMetrics, trainMetrics, false)

    // Convert Json string into Map[String, Any] and check format
    val parsedMetrics = JSON.parseFull(resultString).get.asInstanceOf[Map[String, Any]]

    assertEquals(parsedMetrics.get("HOLD_AUC").get.asInstanceOf[Double], 0.11, 1e-5)
    assertEquals(parsedMetrics.get("TRAIN_ACC").get.asInstanceOf[Double], 0.22, 1e-5)

    assertEquals(parsedMetrics.get("HOLD_THRESHOLDS").get.asInstanceOf[List[Double]].head, 0.1, 1e-5)
    assertEquals(parsedMetrics.get("HOLD_PRECISIONS").get.asInstanceOf[List[Double]].head, 0.2, 1e-5)
    assertEquals(parsedMetrics.get("HOLD_RECALLS").get.asInstanceOf[List[Double]].head, 0.3, 1e-5)

    assertEquals(parsedMetrics.get("TRAIN_THRESHOLDS").get.asInstanceOf[List[Double]].isEmpty, true)
    assertEquals(parsedMetrics.get("TRAIN_PRECISIONS").get.asInstanceOf[List[Double]].isEmpty, true)
    assertEquals(parsedMetrics.get("TRAIN_RECALLS").get.asInstanceOf[List[Double]].isEmpty, true)
  }
}
