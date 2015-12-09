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
  def testMinMax(): Unit = {
    val examples = TrainingTestHelper.makeSimpleClassificationExamples._1
    var sc = new SparkContext("local", "TrainingUtilsTest")

    try {
      val minMax = TrainingUtils.getMinMax(0, sc.parallelize(examples)).toMap
      val minMaxX = minMax.get(("loc", "x")).get
      assertEquals(-1, minMaxX._1, 0.1)
      assertEquals(1, minMaxX._2, 0.1)
      val minMaxY = minMax.get(("loc", "y")).get
      assertEquals(-10.0, minMaxY._1, 1.0)
      assertEquals(10.0, minMaxY._2, 1.0)
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
