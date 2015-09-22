package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core.EvaluationRecord
import org.apache.spark.SparkContext
import org.junit.Test
import org.slf4j.LoggerFactory
import org.junit.Assert.fail
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class EvaluationTest {
  val log = LoggerFactory.getLogger("EvaluationTest")

  def generateDataPerfect(correct : Boolean) = {
    val recs = ArrayBuffer[EvaluationRecord]()
    for (i <- 0 to 1000) {
      for (is_training <- Array(true, false)) {
        val pos = new EvaluationRecord()
        pos.setIs_training(is_training)
        val posScore = if (correct) 1.0 else 0.0
        pos.setScore(posScore)
        pos.setLabel(1.0)
        recs.append(pos)
        val neg = new EvaluationRecord()
        neg.setIs_training(is_training)
        val negScore = if (correct) 0.0 else 1.0
        neg.setScore(negScore)
        neg.setLabel(-1.0)
        recs.append(neg)
      }
    }
    recs
  }

  def generateDataGaussian = {
    val recs = ArrayBuffer[EvaluationRecord]()
    val rnd = new Random(0xDEADBEEF)
    for (i <- 0 to 1000) {
      for (is_training <- Array(true, false)) {
        val pos = new EvaluationRecord()
        pos.setIs_training(is_training)
        val posScore = 1.0 + rnd.nextGaussian()
        pos.setScore(posScore)
        pos.setLabel(1.0)
        recs.append(pos)
        val neg = new EvaluationRecord()
        neg.setIs_training(is_training)
        val negScore = -1.0 + rnd.nextGaussian()
        neg.setScore(negScore)
        neg.setLabel(-1.0)
        recs.append(neg)
      }
    }
    recs
  }

  def generateDataCorrelatedRegression = {
    val recs = ArrayBuffer[EvaluationRecord]()
    val rnd = new Random(0xDEADBEEF)
    for (i <- 0 to 1000) {
      for (is_training <- Array(true, false)) {
        val pos = new EvaluationRecord()
        pos.setIs_training(is_training)
        val posScore = i + rnd.nextGaussian()
        pos.setScore(posScore)
        pos.setLabel(i)
        recs.append(pos)
        val neg = new EvaluationRecord()
        neg.setIs_training(is_training)
        val negScore = -i + rnd.nextGaussian()
        neg.setScore(negScore)
        neg.setLabel(-i)
        recs.append(neg)
      }
    }
    recs
  }


  // The test data is perfectly correlated with the labels
  @Test def evaluationCorrectTest: Unit = {
    val recs = generateDataPerfect(true)
    var sc = new SparkContext("local", "EvaluationTest")

    try {
      val results = Evaluation.evaluateBinaryClassification(sc.parallelize(recs), 11, "!TRAIN_F1").toMap
      results.foreach(res => log.info("%s = %f".format(res._1, res._2)))
      assertEquals(1.0, results.getOrElse("!TRAIN_ACC", 0.0), 0.1)
      assertEquals(1.0, results.getOrElse("!TRAIN_AUC", 0.0), 0.1)
      assertEquals(1.0, results.getOrElse("!TRAIN_PR_AUC", 0.0), 0.1)
      assertEquals(1.0, results.getOrElse("!TRAIN_F1", 0.0), 0.1)
      assertEquals(1.0, results.getOrElse("!TRAIN_RECALL", 0.0), 0.1)
      assertEquals(1.0, results.getOrElse("!TRAIN_PRECISION", 0.0), 0.1)
      assertEquals(0.0, results.getOrElse("!TRAIN_RMSE", 0.0), 0.1)
      assertEquals(0.0, results.getOrElse("!TRAIN_FPR", 0.0), 0.1)
    }
    finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port,
      // since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }

  // The test data is perfectly anti-correlated with the labels
  @Test def evaluationInCorrectTest: Unit = {
    val recs = generateDataPerfect(false)
    var sc = new SparkContext("local", "EvaluationTest")

    try {
      val results = Evaluation.evaluateBinaryClassification(sc.parallelize(recs), 11, "!HOLD_F1").toMap
      results.foreach(res => log.info("%s = %f".format(res._1, res._2)))
      assertEquals(0.0, results.getOrElse("!HOLD_ACC", 0.0), 0.1)
      assertEquals(0.0, results.getOrElse("!HOLD_AUC", 0.0), 0.1)
      assertEquals(0.0, results.getOrElse("!HOLD_PR_AUC", 0.0), 0.1)
      assertEquals(0.0, results.getOrElse("!HOLD_F1", 0.0), 0.1)
      assertEquals(0.0, results.getOrElse("!HOLD_RECALL", 0.0), 0.1)
      assertEquals(0.0, results.getOrElse("!HOLD_PRECISION", 0.0), 0.1)
      assertEquals(1.0, results.getOrElse("!HOLD_RMSE", 0.0), 0.1)
      assertEquals(1.0, results.getOrElse("!HOLD_FPR", 0.0), 0.1)
    }
    finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port,
      // since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }

  @Test def evaluationGaussianTest: Unit = {
    val recs = generateDataGaussian
    var sc = new SparkContext("local", "EvaluationTest")

    try {
      val results = Evaluation.evaluateBinaryClassification(sc.parallelize(recs), 11, "!HOLD_F1").toMap
      results.foreach(res => log.info("%s = %f".format(res._1, res._2)))

      val THRESHOLD = 0.7
      assertTrue(results.getOrElse("!TRAIN_ACC", 0.0) > THRESHOLD)
      assertTrue(results.getOrElse("!TRAIN_AUC", 0.0) > THRESHOLD)
      assertTrue(results.getOrElse("!TRAIN_PR_AUC", 0.0) > THRESHOLD)
      assertTrue(results.getOrElse("!TRAIN_F1", 0.0) > THRESHOLD)
      assertTrue(results.getOrElse("!TRAIN_RECALL", 0.0) > THRESHOLD)
      assertTrue(results.getOrElse("!TRAIN_PRECISION", 0.0) > THRESHOLD)
      assertTrue(results.getOrElse("!HOLD_ACC", 0.0) > THRESHOLD)
      assertTrue(results.getOrElse("!HOLD_AUC", 0.0) > THRESHOLD)
      assertTrue(results.getOrElse("!HOLD_PR_AUC", 0.0) > THRESHOLD)
      assertTrue(results.getOrElse("!HOLD_F1", 0.0) > THRESHOLD)
      assertTrue(results.getOrElse("!HOLD_RECALL", 0.0) > THRESHOLD)
      assertTrue(results.getOrElse("!HOLD_PRECISION", 0.0) > THRESHOLD)
    }
    finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port,
      // since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }

  @Test def evaluationRegressionCorrelatedTest: Unit = {
    val recs = generateDataCorrelatedRegression
    var sc = new SparkContext("local", "EvaluationTest")

    try {
      val results = Evaluation.evaluateRegression(sc.parallelize(recs)).toMap

      val ERROR_THRESHOLD = 2.0
      assertTrue(results.getOrElse("!TRAIN_RMSE", { fail("TRAIN_RMSE missing"); 0.0}) < ERROR_THRESHOLD)
      assertTrue(results.getOrElse("!HOLD_RMSE", { fail("HOLD_RMSE missing"); 0.0}) < ERROR_THRESHOLD)
    } finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port,
      // since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }
}