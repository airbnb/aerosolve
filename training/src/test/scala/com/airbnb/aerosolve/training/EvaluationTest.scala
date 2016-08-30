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
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

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

  def generateMulticlassDataPerfect(positionOfLabel : Int) = {
    val recs = ArrayBuffer[EvaluationRecord]()
    val rnd = new Random(0xDEADBEEF)
    for (i <- 0 to 10) {
      for (is_training <- Array(true, false)) {
        val sample = new EvaluationRecord()
        sample.setIs_training(is_training)

        val scores = new java.util.HashMap[java.lang.String, java.lang.Double]()
        val labels = new java.util.HashMap[java.lang.String, java.lang.Double]()

        scores.put("A", rnd.nextDouble())
        scores.put("B", rnd.nextDouble())
        scores.put("C", rnd.nextDouble())

        val sorted = scores.toBuffer.sortWith((a, b) => a._2 > b._2)
        labels.put(sorted(positionOfLabel)._1, 1.0)

        sample.setScores(scores)
        sample.setLabels(labels)

        recs.append(sample)
      }
    }
    recs
  }

  // The test data is perfectly correlated with the labels
  @Test
  def evaluationCorrectTest(): Unit = {
    val recs = generateDataPerfect(true)
    var sc = new SparkContext("local", "EvaluationTest")

    try {
      val results = Evaluation.evaluateBinaryClassification(sc.parallelize(recs), 11, "!TRAIN_F1").toMap
      results.foreach(res => log.info("%s = %f".format(res._1, res._2)))
      assertEquals(1.0, results.getOrElse("!TRAIN_ACC", 0.0), 0.1)
      assertEquals(1.0, results.getOrElse("!TRAIN_AUC", 0.0), 0.1)
      assertEquals(1.0, results.getOrElse("!TRAIN_PRECISION_RECALL_AUC", 0.0), 0.1)
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
  @Test
  def evaluationInCorrectTest(): Unit = {
    val recs = generateDataPerfect(false)
    var sc = new SparkContext("local", "EvaluationTest")

    try {
      val results = Evaluation.evaluateBinaryClassification(sc.parallelize(recs), 11, "!HOLD_F1").toMap
      results.foreach(res => log.info("%s = %f".format(res._1, res._2)))
      assertEquals(0.0, results.getOrElse("!HOLD_ACC", 0.0), 0.1)
      assertEquals(0.0, results.getOrElse("!HOLD_AUC", 0.0), 0.1)
      assertEquals(0.0, results.getOrElse("!HOLD_PRECISION_RECALL_AUC", 0.0), 0.1)
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

  @Test
  def evaluationInCorrectListTest(): Unit = {
    val recs = generateDataPerfect(false).toList
    var sc = new SparkContext("local", "EvaluationTest")
    try {
      val results1 = Evaluation.evaluateBinaryClassification(recs, 11, "!HOLD_F1").toMap
      val results2 = Evaluation.evaluateBinaryClassification(sc.parallelize(recs), 11, "!HOLD_F1").toMap
      log.info("Non-RDD eval")
      results1.foreach(res => log.info("%s = %f".format(res._1, res._2)))
      log.info("RDD eval")
      results2.foreach(res => log.info("%s = %f".format(res._1, res._2)))

      assertEquals(results1.getOrElse("!HOLD_AUC", 0.0), results2.getOrElse("!HOLD_ACC", 0.0), 0.1)
      assertEquals(results1.getOrElse("!HOLD_PRECISION_RECALL_AUC", 0.0), results2.getOrElse("!HOLD_PRECISION_RECALL_AUC", 0.0), 0.1)
      assertEquals(results1.getOrElse("!HOLD_F1", 0.0), results2.getOrElse("!HOLD_F1", 0.0), 0.1)
      assertEquals(results1.getOrElse("!HOLD_RECALL", 0.0), results2.getOrElse("!HOLD_RECALL", 0.0), 0.1)
      assertEquals(results1.getOrElse("!HOLD_PRECISION", 0.0), results2.getOrElse("!HOLD_PRECISION", 0.0), 0.1)
      assertEquals(results1.getOrElse("!HOLD_RMSE", 0.0), results2.getOrElse("!HOLD_RMSE", 0.0), 0.1)
      assertEquals(results1.getOrElse("!HOLD_FPR", 0.0), results2.getOrElse("!HOLD_FPR", 0.0), 0.1)
    } finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port,
      // since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }

  @Test
  def evaluationGaussianTest(): Unit = {
    val recs = generateDataGaussian
    var sc = new SparkContext("local", "EvaluationTest")

    try {
      val results = Evaluation.evaluateBinaryClassification(sc.parallelize(recs), 11, "!HOLD_F1").toMap
      results.foreach(res => log.info("%s = %f".format(res._1, res._2)))

      val THRESHOLD = 0.7
      assertTrue(results.getOrElse("!TRAIN_ACC", 0.0) > THRESHOLD)
      assertTrue(results.getOrElse("!TRAIN_AUC", 0.0) > THRESHOLD)
      assertTrue(results.getOrElse("!TRAIN_PRECISION_RECALL_AUC", 0.0) > THRESHOLD)
      assertTrue(results.getOrElse("!TRAIN_F1", 0.0) > THRESHOLD)
      assertTrue(results.getOrElse("!TRAIN_RECALL", 0.0) > THRESHOLD)
      assertTrue(results.getOrElse("!TRAIN_PRECISION", 0.0) > THRESHOLD)
      assertTrue(results.getOrElse("!HOLD_ACC", 0.0) > THRESHOLD)
      assertTrue(results.getOrElse("!HOLD_AUC", 0.0) > THRESHOLD)
      assertTrue(results.getOrElse("!HOLD_PRECISION_RECALL_AUC", 0.0) > THRESHOLD)
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

  @Test
  def evaluationGaussianAUCTest(): Unit = {
    val recs = generateDataGaussian
    var sc = new SparkContext("local", "EvaluationTest")

    try {
      val auc = Evaluation.getClassificationAUC(recs.toList)
      val THRESHOLD = 0.7
      assertTrue(auc > THRESHOLD)
    }
    finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port,
      // since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }

  @Test
  def evaluationGaussianListTest(): Unit = {
    val recs = generateDataGaussian.toList
    val results = Evaluation.evaluateBinaryClassification(recs, 11, "!HOLD_F1").toMap
    results.foreach(res => log.info("%s = %f".format(res._1, res._2)))

    val THRESHOLD = 0.7
    assertTrue(results.getOrElse("!TRAIN_ACC", 0.0) > THRESHOLD)
    assertTrue(results.getOrElse("!TRAIN_AUC", 0.0) > THRESHOLD)
    assertTrue(results.getOrElse("!TRAIN_PRECISION_RECALL_AUC", 0.0) > THRESHOLD)
    assertTrue(results.getOrElse("!TRAIN_F1", 0.0) > THRESHOLD)
    assertTrue(results.getOrElse("!TRAIN_RECALL", 0.0) > THRESHOLD)
    assertTrue(results.getOrElse("!TRAIN_PRECISION", 0.0) > THRESHOLD)
    assertTrue(results.getOrElse("!HOLD_ACC", 0.0) > THRESHOLD)
    assertTrue(results.getOrElse("!HOLD_AUC", 0.0) > THRESHOLD)
    assertTrue(results.getOrElse("!HOLD_PRECISION_RECALL_AUC", 0.0) > THRESHOLD)
    assertTrue(results.getOrElse("!HOLD_F1", 0.0) > THRESHOLD)
    assertTrue(results.getOrElse("!HOLD_RECALL", 0.0) > THRESHOLD)
    assertTrue(results.getOrElse("!HOLD_PRECISION", 0.0) > THRESHOLD)
  }

  @Test
  def evaluationRegressionCorrelatedTest(): Unit = {
    val recs = generateDataCorrelatedRegression
    var sc = new SparkContext("local", "EvaluationTest")

    try {
      val results = Evaluation.evaluateRegression(sc.parallelize(recs)).toMap

      val ERROR_THRESHOLD = 2.0
      assertTrue(results.getOrElse("!TRAIN_RMSE", { fail("TRAIN_RMSE missing"); 0.0}) < ERROR_THRESHOLD)
      assertTrue(results.getOrElse("!TRAIN_SMAPE",
        { fail("TRAIN_SMAPE missing"); 0.0}) < ERROR_THRESHOLD)
      assertTrue(results.getOrElse("!HOLD_RMSE", { fail("HOLD_RMSE missing"); 0.0}) < ERROR_THRESHOLD)
      assertTrue(results.getOrElse("!HOLD_SMAPE",
        { fail("HOLD_SMAPE missing"); 0.0}) < ERROR_THRESHOLD)
    } finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port,
      // since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }

  @Test
  def evaluationMulticlassTest(): Unit = {
    for (posOfLabel <- 0 until 3) {
      log.info("Labels at position " + posOfLabel)
      val recs = generateMulticlassDataPerfect(posOfLabel)
      recs.foreach(x => log.info(x.toString))
      var sc = new SparkContext("local", "EvaluationTest")

      try {
        val results = Evaluation.evaluateMulticlassClassification(sc.parallelize(recs)).toMap
        results.foreach(res => log.info("%s = %f".format(res._1, res._2)))
        val expectedA1 = if (posOfLabel == 0) 1.0 else 0.0
        val expectedA2 = if (posOfLabel <= 1) 1.0 else 0.0
        val expectedA3 = 1.0

        val expectedP1 = if (posOfLabel == 0) 1.0 else 0.0
        val expectedP2 = if (posOfLabel <= 1) 0.5 else 0.0
        val expectedP3 = 1.0/3

        val expectedMAP1 = if (posOfLabel == 0) 1.0 else 0.0
        val expectedMAP2 = posOfLabel match {
          case 0 => 1.0
          case 1 => 0.5
          case 2 => 0.0
        }
        val expectedMAP3 = posOfLabel match {
          case 0 => 1.0
          case 1 => 0.5
          case 2 => 1.0/3
        }

        assertEquals(expectedA1, results.getOrElse("TRAIN_ACCURACY@1", 0.0), 0.1)
        assertEquals(expectedA2, results.getOrElse("TRAIN_ACCURACY@2", 0.0), 0.1)
        assertEquals(expectedA3, results.getOrElse("TRAIN_ACCURACY@3", 0.0), 0.1)
        assertEquals(expectedA1, results.getOrElse("HOLD_ACCURACY@1", 0.0), 0.1)
        assertEquals(expectedA2, results.getOrElse("HOLD_ACCURACY@2", 0.0), 0.1)
        assertEquals(expectedA3, results.getOrElse("HOLD_ACCURACY@3", 0.0), 0.1)

        assertEquals(expectedP1, results.getOrElse("TRAIN_PRECISION@1", 0.0), 0.1)
        assertEquals(expectedP2, results.getOrElse("TRAIN_PRECISION@2", 0.0), 0.1)
        assertEquals(expectedP3, results.getOrElse("TRAIN_PRECISION@3", 0.0), 0.1)
        assertEquals(expectedP1, results.getOrElse("HOLD_PRECISION@1", 0.0), 0.1)
        assertEquals(expectedP2, results.getOrElse("HOLD_PRECISION@2", 0.0), 0.1)
        assertEquals(expectedP3, results.getOrElse("HOLD_PRECISION@3", 0.0), 0.1)

        assertEquals(expectedMAP1, results.getOrElse("TRAIN_MEAN_AVERAGE_PRECISION@1", 0.0), 0.1)
        assertEquals(expectedMAP2, results.getOrElse("TRAIN_MEAN_AVERAGE_PRECISION@2", 0.0), 0.1)
        assertEquals(expectedMAP3, results.getOrElse("TRAIN_MEAN_AVERAGE_PRECISION@3", 0.0), 0.1)
        assertEquals(expectedMAP1, results.getOrElse("HOLD_MEAN_AVERAGE_PRECISION@1", 0.0), 0.1)
        assertEquals(expectedMAP2, results.getOrElse("HOLD_MEAN_AVERAGE_PRECISION@2", 0.0), 0.1)
        assertEquals(expectedMAP3, results.getOrElse("HOLD_MEAN_AVERAGE_PRECISION@3", 0.0), 0.1)

        assertEquals(1.0 / (posOfLabel + 1.0), results.getOrElse("TRAIN_MEAN_RECIPROCAL_RANK", 0.0), 0.1)
        assertEquals(1.0 / (posOfLabel + 1.0), results.getOrElse("HOLD_MEAN_RECIPROCAL_RANK", 0.0), 0.1)
        assertTrue(results.getOrElse("TRAIN_ALL_PAIRS_HINGE_LOSS", 0.0) > posOfLabel)
        assertTrue(results.getOrElse("HOLD_ALL_PAIRS_HINGE_LOSS", 0.0) > posOfLabel)
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
}
