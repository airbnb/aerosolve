package com.airbnb.aerosolve.training

import org.apache.spark.SparkContext
import org.junit.Test
import org.slf4j.LoggerFactory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue

class HistogramCalibratorTest {
  val log = LoggerFactory.getLogger("HistogramCalibratorTest")

  @Test
  def testCalibrate = {
    var sc = new SparkContext("local", "HistogramCalibratorTest")
    val src = sc.parallelize(6 to 99)
    val square = src.map(x => 1.0 * x * x)
    val calibrator = HistogramCalibrator(square, 16)
    calibrator.cumulativeDistributionFunction.foreach(
      x => log.info("%f,%f".format(x._1, x._2))
    )
    // Check the extrema
    assertEquals(0.0, calibrator.cumulativeDistributionFunction.head._2, 1e-2)
    assertEquals(1.0, calibrator.cumulativeDistributionFunction.last._2, 1e-2)
    assertEquals(0.0, calibrator.calibrateScore(5 * 5), 1e-2)
    assertEquals(0.0, calibrator.calibrateScore(6 * 6), 1e-2)
    assertEquals(1.0, calibrator.calibrateScore(99 * 99), 1e-2)
    assertEquals(1.0, calibrator.calibrateScore(100 * 100), 1e-2)
    // Check the quartiles
    val median = (6 + 99) / 2.0
    assertEquals(0.5, calibrator.calibrateScore(median * median), 1e-2)
    val lowerQuartile = 6.0 + (99 - 6) * 0.25
    assertEquals(0.25, calibrator.calibrateScore(lowerQuartile * lowerQuartile), 1e-2)
    val upperQuartile = 6.0 + (99 - 6) * 0.75
    assertEquals(0.75, calibrator.calibrateScore(upperQuartile * upperQuartile), 1e-2)
    // Check the thirds
    val lowerThird = 6.0 + (99 - 6) / 3.0
    assertEquals(0.33, calibrator.calibrateScore(lowerThird * lowerThird), 1e-2)
    val secondThird = 6.0 + 2.0 * (99 - 6) / 3.0
    assertEquals(0.66, calibrator.calibrateScore(secondThird * secondThird), 1e-2)
    // Check linear interpolation
    for (i <- 7 until 98) {
      val score = i * i * 1.0
      assertTrue(calibrator.calibrateScore(score) < calibrator.calibrateScore(score + 1.0))
    }

    try {
   } finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }
}