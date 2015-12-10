package com.airbnb.aerosolve.training

import com.typesafe.config.ConfigFactory
import org.slf4j.LoggerFactory
import org.junit.Test
import org.apache.spark.SparkContext
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import com.airbnb.aerosolve.core.util.Util

class TrainingUtilsTest {
  val log = LoggerFactory.getLogger("TrainingUtilsTest")

  @Test
  def testFeatureStatistics(): Unit = {
    val examples = TrainingTestHelper.makeSimpleClassificationExamples._1
    var sc = new SparkContext("local", "TrainingUtilsTest")

    try {
      val statsArr = TrainingUtils.getFeatureStatistics(0, sc.parallelize(examples))
      val stats = statsArr.toMap
      val statsX = stats.get(("loc", "x")).get
      assertEquals(-1, statsX.min, 0.1)
      assertEquals(1, statsX.max, 0.1)
      assertEquals(0, statsX.mean, 0.1)
      // X is uniform between -1 and 1
      val xvar = 2.0 * 2.0 / 12.0
      assertEquals(xvar, statsX.variance, 0.1)
      val statsY = stats.get(("loc", "y")).get
      assertEquals(-10.0, statsY.min, 1.0)
      assertEquals(10.0, statsY.max, 1.0)
      assertEquals(0.0, statsY.mean, 1.0)
      // Y is uniform between -10 and 10
      val yvar = 20.0 * 20.0 / 12.0
      assertEquals(yvar, statsY.variance, 1.0)
      val statsBias = stats.get(("BIAS", "B")).get
      assertEquals(1.0, statsBias.min, 0.1)
      assertEquals(1.0, statsBias.max, 0.1)
      assertEquals(1.0, statsBias.mean, 0.1)
      assertEquals(0.0, statsBias.variance, 0.1)
      val statsNeg = stats.get(("NEG", "T")).get
      assertEquals(1.0, statsNeg.min, 0.1)
      assertEquals(1.0, statsNeg.max, 0.1)
      assertEquals(1.0, statsNeg.mean, 0.1)
      assertEquals(0.0, statsNeg.variance, 0.1)

      val dictionary = TrainingUtils.createStringDictionaryFromFeatureStatistics(statsArr, Set("$rank"))
      assertEquals(4, dictionary.getDictionary().getEntryCount())
      val ex = TrainingTestHelper.makeExample(2.0, 1.0, 2)
      log.info(ex.toString)
      val vec = dictionary.makeVectorFromSparseFloats(Util.flattenFeature(ex.example.get(0)));
      assertEquals(4, vec.values.length);
      log.info(vec.toString)
      val arr = scala.collection.mutable.ArrayBuffer[Float]()
      for (v <- vec.values) {
        arr += v
      }
      val sorted = arr.sortWith((a, b) => a < b)
      // Neg is missing
      assertEquals(0.0, sorted(0), 0.2)
      // Y
      assertEquals(1.0 / Math.sqrt(yvar), sorted(1), 0.2)
      // Bias
      assertEquals(1.0, sorted(2), 0.1)
      // X
      assertEquals(2.0 / Math.sqrt(xvar), sorted(3), 0.2)
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
