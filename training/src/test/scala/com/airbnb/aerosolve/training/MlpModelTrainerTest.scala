package com.airbnb.aerosolve.training

import java.io.{StringReader, BufferedWriter, BufferedReader, StringWriter}

import com.airbnb.aerosolve.core.models.{ModelFactory, MlpModel}
import com.airbnb.aerosolve.core.Example
import com.typesafe.config.ConfigFactory
import org.apache.spark.SparkContext
import org.junit.Test
import org.slf4j.LoggerFactory
import org.junit.Assert._
import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._

class MlpModelTrainerTest {
  val log = LoggerFactory.getLogger("MlpModelTrainerTest")
  def makeConfig(dropout : Double,
                 momentumT : Int,
                 loss : String,
                 extraArgs : String,
                 weightDecay : Double = 0.0,
                 margin : Double = 1.0,
                 learningRateInit: Double = 0.1) : String = {
    """
      |identity_transform {
      |  transform : list
      |  transforms: []
      |}
      |model_config {
      |  %s
      |  loss : %s
      |  rank_key : "$rank"
      |  rank_threshold : 0.0
      |  margin : %f
      |  learning_rate_init : %f
      |  learning_rate_decay : 0.95
      |  momentum_init : 0.5
      |  momentum_end : 0.9
      |  momentum_t : %d
      |  weight_decay : %f
      |  weight_init_std : 0.5
      |  iterations : 50
      |  dropout : %f
      |  min_count : 0
      |  subsample : 0.1
      |  cache : "cache"
      |  context_transform : identity_transform
      |  item_transform : identity_transform
      |  combined_transform : identity_transform
      |  activations : ["tanh", "identity"]
      |  node_number : [5, 1]
      |  model_output : ""
      |}
    """.stripMargin.format(extraArgs, loss, margin, learningRateInit, momentumT, weightDecay, dropout)
  }

  // TODO (peng): add more tests and gradient checks
  @Test
  def testModelTrainerHingeNonLinear() : Unit = {
    testMlpModelTrainer("hinge", 0.0, "", 0, 0.0, "poly")
  }

  @Test
  def testModelTrainerHingeLinear() : Unit = {
    testMlpModelTrainer("hinge", 0.0, "", 0, 0.0, "linear")
  }

  @Test
  def testModelTrainerHingeNonLinearWithDropout() : Unit = {
    testMlpModelTrainer("hinge", 0.1, "", 0, 0.0, "poly")
  }

  @Test
  def testModelTrainerHingeLinearWithDropout() : Unit = {
    testMlpModelTrainer("hinge", 0.1, "", 0, 0.0, "linear")
  }

  @Test
  def testModelTrainerHingeNonLinearWithMomentum() : Unit = {
    testMlpModelTrainer("hinge", 0.0, "", 50, 0.0, "poly")
  }

  @Test
  def testModelTrainerHingeLinearWithMomentum() : Unit = {
    testMlpModelTrainer("hinge", 0.0, "", 50, 0.0, "linear")
  }

  @Test
  def testModelTrainerHingeNonLinearWithWeightDecay() : Unit = {
    testMlpModelTrainer("hinge", 0.0, "", 0,  weightDecay = 0.0001, "poly")
  }

  @Test
  def testModelTrainerHingeLinearWithWeightDecay() : Unit = {
    testMlpModelTrainer("hinge", 0.0, "", 0, 0.0001, "linear")
  }

  @Test
  def testRegression(): Unit = {
    testRegressionModel(0.0, "", 0, weightDecay = 0.0, epsilon = 0.1, learningRateInit = 0.2)
  }

  @Test
  def testRegressionWithDropout(): Unit = {
    testRegressionModel(0.1, "", 0, weightDecay = 0.01, epsilon = 0.1, learningRateInit = 0.2)
}

  def testMlpModelTrainer(loss : String,
                          dropout : Double,
                          extraArgs : String,
                          momentumT : Int,
                          weightDecay : Double = 0.0,
                          exampleFunc: String = "poly") = {
    var sc = new SparkContext("local", "MlpModelTrainerTest")
    try {
      val (examples, label, numPos) = if (exampleFunc.equals("poly")) {
        TrainingTestHelper.makeClassificationExamples
      } else {
        TrainingTestHelper.makeLinearClassificationExamples
      }
      val config = ConfigFactory.parseString(makeConfig(dropout, momentumT, loss, extraArgs, weightDecay))
      val input = sc.parallelize(examples)
      val model = MlpModelTrainer.train(sc, input, config, "model_config")
      testClassificationModel(model, examples, label, numPos)
    } finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }
  def testClassificationModel(model: MlpModel,
                              examples: ArrayBuffer[Example],
                              label: ArrayBuffer[Double],
                              numPos: Int): Unit = {
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
    assertTrue(fracCorrect > 0.7)

    val swriter = new StringWriter()
    val writer = new BufferedWriter(swriter)
    model.save(writer)
    writer.close()
    val str = swriter.toString
    val sreader = new StringReader(str)
    val reader = new BufferedReader(sreader)
    log.info(str)
    val model2Opt = ModelFactory.createFromReader(reader)
    assertTrue(model2Opt.isPresent)
    val model2 = model2Opt.get()
    for (ex <- examples) {
      val score = model.scoreItem(ex.example.get(0))
      val score2 = model2.scoreItem(ex.example.get(0))
      assertEquals(score, score2, 0.01f)
    }
  }

  def testRegressionModel(dropout : Double,
                          extraArgs : String,
                          momentumT : Int,
                          weightDecay : Double,
                          epsilon: Double = 0.1,
                          learningRateInit: Double = 0.1): Unit = {
    val (trainingExample, trainingLabel) = TrainingTestHelper.makeRegressionExamples()
    var sc = new SparkContext("local", "MlpRegressionTest")
    try {
      val config = ConfigFactory.parseString(makeConfig(
        dropout, momentumT, "regression", extraArgs, weightDecay = weightDecay,
        margin = epsilon, learningRateInit = learningRateInit))
      val input = sc.parallelize(trainingExample)
      val model = MlpModelTrainer.train(sc, input, config, "model_config")
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
      // compute testing error
      val (testingExample, testingLabel) = TrainingTestHelper.makeRegressionExamples(25)
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
      log.info("Training: Average absolute error = %f".format(trainError))
      log.info("Testing: Average absolute error = %f".format(testError))

      assertTrue(trainError < 3.0)
      assertTrue(testError < 4.0)
    } finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }
}
