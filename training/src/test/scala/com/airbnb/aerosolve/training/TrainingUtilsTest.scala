package com.airbnb.aerosolve.training

import com.typesafe.config.ConfigFactory
import org.slf4j.LoggerFactory
import org.junit.Test
import org.apache.spark.SparkContext
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue

class TrainingUtilsTest {
  val log = LoggerFactory.getLogger("TrainingUtilsTest")

  @Test
  def testFeatureStatistics(): Unit = {
    val examples = TrainingTestHelper.makeSimpleClassificationExamples._1
    var sc = new SparkContext("local", "TrainingUtilsTest")

    try {
      val stats = TrainingUtils.getFeatureStatistics(0, sc.parallelize(examples)).toMap
      val statsX = stats.get(("loc", "x")).get
      assertEquals(-1, statsX.min, 0.1)
      assertEquals(1, statsX.max, 0.1)
      assertEquals(0, statsX.mean, 0.1)
      assertEquals(2.0 * 2.0 / 12.0, statsX.variance, 0.1)
      val statsY = stats.get(("loc", "y")).get
      assertEquals(-10.0, statsY.min, 1.0)
      assertEquals(10.0, statsY.max, 1.0)
      assertEquals(0.0, statsY.mean, 1.0)
      assertEquals(20.0 * 20.0 / 12.0, statsY.variance, 1.0)
    }
    finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port,
      // since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }

}
