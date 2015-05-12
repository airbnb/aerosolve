package com.airbnb.aerosolve.training

import org.junit.Test
import org.slf4j.LoggerFactory
import org.junit.Assert.assertEquals

class SplineQualityMetricsTest {
  val log = LoggerFactory.getLogger("SplineQualityMetricsTest")

  def makeData(monotonic : String, direction: String) : Array[Double] = {
      (monotonic, direction) match {
        case ("monotonic", "increasing") => (1 to 5).map(x => x.toDouble).toArray
        case ("monotonic", "decreasing") => (0 to 4).map(x => 5 - x.toDouble).toArray
        case ("nonmonotonic", "increasing") => Array[Double](1.0, 3.0, 2.0, 4.0, 5.0)
        case ("nonmonotonic", "decreasing") => Array[Double](5.0, 3.0, 4.0, 2.0, 1.0)
      }
  }

  @Test
  def testSplineQualityMetricsSmoothDecreasing : Unit = {
    testSplineQualityMetricsSmoothness("monotonic", "decreasing")
    testSplineQualityMetricsMonotonicity("monotonic", "decreasing")
  }

  @Test
  def testSplineQualityMetricsSmoothIncreasing : Unit = {
    testSplineQualityMetricsSmoothness("monotonic", "increasing")
    testSplineQualityMetricsMonotonicity("monotonic", "increasing")
  }

  @Test
  def testSplineQualityMetricsNonSmoothDecreasing : Unit = {
    testSplineQualityMetricsSmoothness("nonmonotonic", "decreasing")
    testSplineQualityMetricsMonotonicity("nonmonotonic", "decreasing")
  }

  @Test
  def testSplineQualityMetricsNonSmoothIncreasing : Unit = {
    testSplineQualityMetricsSmoothness("nonmonotonic", "increasing")
    testSplineQualityMetricsMonotonicity("nonmonotonic", "increasing")
  }

  def testSplineQualityMetricsMonotonicity(monotonic: String,
                                           direction: String) = {
    val data = makeData(monotonic, direction)

    if (monotonic == "monotonic"){
      assertEquals(1.0, SplineQualityMetrics.getMonotonicity(data), 0.01)
    } else if (monotonic == "nonmonotonic") {
      assertEquals(0.8, SplineQualityMetrics.getMonotonicity(data), 0.01)
    }
  }

  def testSplineQualityMetricsSmoothness(monotonic: String,
                                           direction: String) = {
    val data = makeData(monotonic, direction)

    if (monotonic == "monotonic"){
      assertEquals(1.0, SplineQualityMetrics.getSmoothness(data), 0.01)
    } else if (monotonic == "nonmonotonic") {
      assertEquals(0.4, SplineQualityMetrics.getSmoothness(data), 0.01)
    }
  }



}