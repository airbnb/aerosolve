package com.airbnb.aerosolve.training

import com.typesafe.config.ConfigFactory
import org.slf4j.LoggerFactory
import org.junit.Test
import org.apache.spark.SparkContext


class ScoreCalibratorTest {
  val log = LoggerFactory.getLogger("ScoreCalibratorTest")

  def makeConfigBatch: String = {
    """
      |model_config {
      |  iterations : 500
      |  learning_rate : 10.0
      |  rate_decay : 0.95
      |  tolerance : 0.00001
      |}
    """.stripMargin
  }

  def makeConfigSGD: String = {
    """
      |model_config {
      |  iterations : 500
      |  learning_rate : 0.1
      |  rate_decay : 0.95
      |  num_bags : 10
      |  tolerance : 0.0001
      |}
    """.stripMargin
  }

  @Test
  def testScoreCalibrator(): Unit = {
    // generate random number in the range of (min_score, max_score) as scores
    val num = 30000
    val a = -1.5
    val b = 2

    val r = new scala.util.Random(num * 2)
    // generate random number in the range (0, 1)
    val data = Array.fill(num)(r.nextDouble())
    val prob = data.map(x => 1.0 / (1.0 + math.exp(- a - b * x))) // probability of being positive
    val trainData = data.zip(prob).map(x => (x._1, if (r.nextDouble() <= x._2) true else false))

    var sc = new SparkContext("local", "ScoreCaliboratorTest")

    try {
      testScoreCalibratorRun(sc, trainData, "SGD", a, b)
      testScoreCalibratorRun(sc, trainData, "Batch", a, b)
    }
    finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port,
      // since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }

  def testScoreCalibratorRun(sc : SparkContext,
                             trainData : Array[(Double, Boolean)],
                             trainType : String,
                             offset : Double,
                             slope : Double): Unit = {

    val trainRDD = sc.parallelize(trainData)
    val config = trainType match {
      case "Batch" => ConfigFactory.parseString(makeConfigBatch)
      case "SGD" => ConfigFactory.parseString(makeConfigSGD)
    }

    val result = trainType match {
      case "Batch" => ScoreCalibrator.trainBatchMLE(config.getConfig("model_config"), trainData)
      case "SGD" => ScoreCalibrator.trainSGD(config.getConfig("model_config"), trainRDD)
      }

    assert(result.nonEmpty)

    log.info("offset = %f, slope = %f".format(result(0), result(1)))
    val errorOffset = math.abs(offset - result(0))
    val errorSlope = math.abs(slope - result(1))
    log.info("Absolute error: offset %f, slope %f".format(errorOffset, errorSlope))
    log.info("Relative error: offset %f, slope %f".format(errorOffset / math.abs(offset), errorSlope / math.abs(slope)))
    assert(errorOffset / math.abs(offset) < 0.1 && errorSlope / math.abs(slope) < 0.1)
    }
}
