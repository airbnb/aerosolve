package com.airbnb.aerosolve.training

import java.io.{StringReader, BufferedWriter, BufferedReader, StringWriter}

import com.airbnb.aerosolve.core.models.{ModelFactory, AdditiveModel}
import com.airbnb.aerosolve.core.Example
import com.typesafe.config.ConfigFactory
import org.apache.spark.SparkContext
import org.junit.Test
import org.slf4j.LoggerFactory
import org.junit.Assert._
import scala.collection.mutable.ArrayBuffer

class AdditiveModelTrainerTest {
  val log = LoggerFactory.getLogger("AdditiveModelTrainerTest")
  def makeConfig(loss : String, dropout : Double, extraArgs : String) : String = {
    """
      |identity_transform {
      |  transform : list
      |  transforms: []
      |}
      |model_config {
      |  num_bags : 3
      |  loss : "%s"
      |  %s
      |  rank_key : "$rank"
      |  rank_threshold : 0.0
      |  learning_rate : 0.1
      |  num_bins : 16
      |  iterations : 10
      |  smoothing_tolerance : 0.1
      |  linfinity_threshold : 0.01
      |  linfinity_cap : 10.0
      |  dropout : %f
      |  min_count : 0
      |  subsample : 1.0
      |  context_transform : identity_transform
      |  item_transform : identity_transform
      |  combined_transform : identity_transform
      |  model_output : ""
      |}
    """.stripMargin.format(loss, extraArgs, dropout)
  }

  def makeRegressionConfig(extraArgs: String) : String = {
    """
      |identity_transform {
      |  transform : list
      |  transforms: []
      |}
      |model_config {
      |  num_bags : 3
      |  loss : "regression"
      |  %s
      |  rank_key : "$rank"
      |  rank_threshold : 0.0
      |  learning_rate : 0.1
      |  num_bins : 16
      |  iterations : 10
      |  smoothing_tolerance : 0.1
      |  linfinity_threshold : 0.01
      |  linfinity_cap : 10.0
      |  dropout : 0.0
      |  min_count : 0
      |  subsample : 1.0
      |  epsilon: 0.1
      |  context_transform : identity_transform
      |  item_transform : identity_transform
      |  combined_transform : identity_transform
      |  model_output : ""
      |}
    """.stripMargin.format(extraArgs)
  }

  @Test
  def testAdditiveModelTrainerSplineLogistic : Unit = {
    testAdditiveModelTrainer("logistic", 0.0, "")
  }

  @Test
  def testAdditiveModelTrainerSplineHinge : Unit = {
    testAdditiveModelTrainer("hinge", 0.0, "")
  }

  @Test
  def testAdditiveModelTrainerSplineLogisticWithDropout : Unit = {
    testAdditiveModelTrainer("logistic", 0.2, "")
  }

  @Test
  def testAdditiveModelTrainerSplineHingeWithMargin : Unit = {
    testAdditiveModelTrainer("hinge", 0.0, "margin : 0.5")
  }

  @Test
  def testAdditiveTrainerSplineHingeMultiscale : Unit = {
    testAdditiveModelTrainer("hinge", 0.0, "multiscale : [5, 7, 16]")
  }

  @Test
  def testAdditiveTrainerSplineHingeMultiscaleWithMargin : Unit = {
    testAdditiveModelTrainer("hinge", 0.0, "margin : 0.5, multiscale : [5, 7, 16]")
  }

  @Test
  def testAdditiveModelTrainerLinearHinge1 : Unit = {
    testAdditiveModelTrainer("hinge", 0.0, "", "linear")
  }

  @Test
  def testAdditiveModelTrainerLinearHinge2 : Unit = {
  testAdditiveModelTrainer("hinge", 0.0, "linear_feature:[loc, xy]", "linear")
  }

  @Test
  def testAdditiveModelTrainerLinearLogistic : Unit = {
    testAdditiveModelTrainer("logistic", 0.0, "linear_feature:[loc, xy]", "linear")
  }

  @Test
  def testAdditiveModelTrainerLinearLogisticWithDropout : Unit = {
    testAdditiveModelTrainer("logistic", 0.1, "linear_feature:[loc, xy]", "linear")
  }

  @Test
  def testAdditiveModelTrainerLinearHingeWithMargin : Unit = {
    testAdditiveModelTrainer("hinge", 0.0, "margin:0.5, linear_feature:[loc, xy]", "linear")
  }

  @Test
  def testAdditiveModelTrainerHybridHinge : Unit = {
    testAdditiveModelTrainer("hinge", 0.0, "linear_feature:[xy]", "linear")
  }

  @Test
  def testAdditiveModelTrainerHybridLogistic : Unit = {
    testAdditiveModelTrainer("logistic", 0.0, "linear_feature:[loc]", "linear")
  }

  @Test
  def testAdditiveModelTrainerHybridLogisticWithDropout : Unit = {
    testAdditiveModelTrainer("logistic", 0.1, "linear_feature:[loc]", "linear")
  }

  @Test
  def testAdditiveModelTrainerHybridHingeWithMargin : Unit = {
    testAdditiveModelTrainer("hinge", 0.0, "margin:0.5, linear_feature:[xy]", "linear")
  }

  @Test
  def testAdditiveModelRegressionSpline1: Unit = {
    testAdditiveModelTrainerRegression("", "flattenedQuadratic")
  }

  @Test
  def testAdditiveModelRegressionSpline2: Unit = {
    testAdditiveModelTrainerRegression("", "linear")
  }

  @Test
  def testAdditiveModelRegressionLinear: Unit = {
    testAdditiveModelTrainerRegression("linear_feature:[loc, xy]", "linear")
  }

  @Test
  def testAdditiveModelRegressionLinearMultiscale: Unit = {
    testAdditiveModelTrainerRegression("linear_feature:[loc, xy], multiscale : [5, 7, 16]", "linear")
  }

  @Test
  def testAdditiveModelRegressionHybrid1: Unit = {
    testAdditiveModelTrainerRegression("linear_feature:[loc]", "linear")
  }

  @Test
  def testAdditiveModelRegressionHybridMultiscale1: Unit = {
    testAdditiveModelTrainerRegression("linear_feature:[loc], multiscale : [5, 7, 16]", "linear")
  }

  @Test
  def testAdditiveModelRegressionHybrid2: Unit = {
    testAdditiveModelTrainerRegression("linear_feature:[xy]", "linear")
  }

  @Test
  def testAdditiveModelRegressionHybridMultiscale2: Unit = {
    testAdditiveModelTrainerRegression("linear_feature:[xy], multiscale : [5, 7, 16]", "linear")
  }

  def testAdditiveModelTrainer(loss : String, dropout : Double, extraArgs : String, exampleFunc: String = "poly") = {
    var sc = new SparkContext("local", "AdditiveModelTest")
    try {
      val (examples, label, numPos) = if (exampleFunc.equals("poly")) {
      TrainingTestHelper.makeClassificationExamples
      } else {
      TrainingTestHelper.makeLinearClassificationExamples
      }
      val config = ConfigFactory.parseString(makeConfig(loss, dropout, extraArgs))
      val input = sc.parallelize(examples)
      val model = AdditiveModelTrainer.train(sc, input, config, "model_config")
      testClassificationModel(model, examples, label, numPos)
    } finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }

  def testAdditiveModelTrainerRegression(extraArgs: String, exampleFunc: String) = {
    var sc = new SparkContext("local", "AdditiveModelTest")
    try {
      val (examples, label) = if (exampleFunc.equals("linear")) {
        TrainingTestHelper.makeLinearRegressionExamples()
      } else {
        TrainingTestHelper.makeRegressionExamples()
      }

      val (testingExample, testingLabel) = if (exampleFunc.equals("linear")) {
        TrainingTestHelper.makeLinearRegressionExamples(25)
      } else {
        TrainingTestHelper.makeRegressionExamples(25)
      }

      val threshold = if (exampleFunc.equals("linear")) {
        0.5
      } else {
        3.0
      }

      val config = ConfigFactory.parseString(makeRegressionConfig(extraArgs))
      val input = sc.parallelize(examples)
      val model = AdditiveModelTrainer.train(sc, input, config, "model_config")

      testRegressionModel(model, examples, label, testingExample, testingLabel, threshold)
    } finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }

  def testClassificationModel(model: AdditiveModel,
                              examples: ArrayBuffer[Example],
                              label: ArrayBuffer[Double],
                              numPos: Int): Unit = {
    TrainingTestHelper.printAdditiveModel(model)
    var numCorrect : Int = 0
    var i : Int = 0
    val labelArr = label.toArray
    for (ex <- examples) {
      val score = model.scoreItem(ex.example.get(0))
      if (score * labelArr(i) > 0) {
        numCorrect += 1
      }
      i += 1
    }
    val fracCorrect : Double = numCorrect * 1.0 / examples.length
    log.info("Num correct = %d, frac correct = %f, num pos = %d, num neg = %d"
               .format(numCorrect, fracCorrect, numPos, examples.length - numPos))
    assertTrue(fracCorrect > 0.6)

    val swriter = new StringWriter()
    val writer = new BufferedWriter(swriter)
    model.save(writer)
    writer.close()
    val str = swriter.toString()
    val sreader = new StringReader(str)
    val reader = new BufferedReader(sreader)
    log.info(str)
    val model2Opt = ModelFactory.createFromReader(reader)
    assertTrue(model2Opt.isPresent())
    val model2 = model2Opt.get()
    for (ex <- examples) {
      val score = model.scoreItem(ex.example.get(0))
      val score2 = model2.scoreItem(ex.example.get(0))
      assertEquals(score, score2, 0.01f)
    }
  }

  def testRegressionModel(model: AdditiveModel,
                          trainingExample: ArrayBuffer[Example],
                          trainingLabel: ArrayBuffer[Double],
                          testingExample: ArrayBuffer[Example],
                          testingLabel: ArrayBuffer[Double],
                          threshold: Double = 3.0): Unit = {
    TrainingTestHelper.printAdditiveModel(model)
    val trainLabelArr = trainingLabel.toArray
    var trainTotalError : Double = 0
    var i = 0
    // compute training error
    for (ex <- trainingExample) {
      val score = model.scoreItem(ex.example.get(0))
      val label = trainLabelArr(i)
      trainTotalError += math.abs(score - label)
      i += 1
    }
    val trainError = trainTotalError / trainingExample.size.toDouble
    log.info("Training: Average absolute error = %f".format(trainError))
    // Total error not too high
    assertTrue(trainError < 3.0)
    // compute testing error
    val testLabelArr = testingLabel.toArray
    var testTotalError : Double = 0
    // compute training error
    i = 0
    for (ex <- testingExample) {
      val score = model.scoreItem(ex.example.get(0))
      val label = testLabelArr(i)
      testTotalError += math.abs(score - label)
      i += 1
    }
    val testError = testTotalError / testingExample.size.toDouble
    log.info("Testing: Average absolute error = %f".format(testError))
    assertTrue(testError < threshold)
  }
}
