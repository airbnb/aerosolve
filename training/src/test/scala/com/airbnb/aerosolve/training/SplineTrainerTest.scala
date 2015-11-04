package com.airbnb.aerosolve.training

import java.io.{StringReader, BufferedWriter, BufferedReader, StringWriter}

import com.airbnb.aerosolve.core.models.SplineModel.WeightSpline
import com.airbnb.aerosolve.core.models.{ModelFactory, SplineModel}
import com.typesafe.config.ConfigFactory
import org.apache.spark.SparkContext
import org.junit.Test
import org.slf4j.LoggerFactory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import scala.collection.JavaConverters._

class SplineTrainerTest {
  val log = LoggerFactory.getLogger("SplineTrainerTest")

  def makeSplineModel() : SplineModel = {
    val model: SplineModel = new SplineModel()
    val weights = new java.util.HashMap[String, java.util.Map[String, WeightSpline]]()
    val innerA = new java.util.HashMap[String, WeightSpline]
    val innerB = new java.util.HashMap[String, WeightSpline]
    val a = new WeightSpline(1.0f, 10.0f, 2)
    val b = new WeightSpline(1.0f, 10.0f, 2)
    a.splineWeights(0) = 1.0f
    a.splineWeights(1) = 2.0f
    b.splineWeights(0) = 1.0f
    b.splineWeights(1) = 5.0f
    weights.put("A", innerA)
    weights.put("B", innerB)
    innerA.put("a", a)
    innerB.put("b", b)
    model.setNumBins(2)
    model.setWeightSpline(weights)
    model.setOffset(0.5f)
    model.setSlope(1.5f)
    return model
  }

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
      |  learning_rate : 0.5
      |  num_bins : 16
      |  iterations : 10
      |  smoothing_tolerance : 0.1
      |  linfinity_threshold : 0.01
      |  linfinity_cap : 1.0
      |  dropout : %f
      |  min_count : 0
      |  subsample : 1.0
      |  context_transform : identity_transform
      |  item_transform : identity_transform
      |  combined_transform : identity_transform
      |}
    """.stripMargin.format(loss, extraArgs, dropout)
  }

  def makeRegressionConfig() : String = {
    """
      |identity_transform {
      |  transform : list
      |  transforms: []
      |}
      |model_config {
      |  num_bags : 3
      |  loss : "regression"
      |  rank_key : "$rank"
      |  rank_threshold : 0.0
      |  learning_rate : 0.5
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
      |}
    """.stripMargin
  }

  @Test
  def testSplineTrainerLogistic : Unit = {
    testSplineTrainer("logistic", 0.0, "")
  }

  @Test
  def testSplineTrainerHinge : Unit = {
    testSplineTrainer("hinge", 0.0, "")
  }
  
  @Test
  def testSplineTrainerLogisticWithDropout : Unit = {
    testSplineTrainer("logistic", 0.2, "")
  }

  @Test
  def testSplineTrainerHingeWithDropout : Unit = {
    testSplineTrainer("hinge", 0.2, "")
  }
  
  @Test
  def testSplineTrainerHingeWithMargin : Unit = {
    testSplineTrainer("hinge", 0.0, "margin : 0.5")
  }
  
  
  @Test
  def testSplineTrainerLogisticMultiscale : Unit = {
    testSplineTrainer("logistic", 0.0, "multiscale : [5, 7, 16]")
  }
  
  @Test
  def testSplineTrainerHingeMultiscale : Unit = {
    testSplineTrainer("hinge", 0.0, "multiscale : [ 5, 7, 16]")
  }
  
  
  @Test
  def testSplineTrainerHingeMultiscaleLessBags : Unit = {
    testSplineTrainer("hinge", 0.2, "multiscale : [ 7, 16]")
  }

  def testSplineTrainer(loss : String, dropout : Double, extraArgs : String) = {
    val (examples, label, numPos) = TrainingTestHelper.makeClassificationExamples

    var sc = new SparkContext("local", "SplineTest")

    try {
      val config = ConfigFactory.parseString(makeConfig(loss, dropout, extraArgs))

      val input = sc.parallelize(examples)
      val model = SplineTrainer.train(sc, input, config, "model_config")

      TrainingTestHelper.printSpline(model)

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

   } finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }

  @Test
  def testAddSpline(): Unit = {
    val model = makeSplineModel()
    // add an existing feature without overwrite
    model.addSpline("A", "a", 0.0f, 1.0f, false)
    // add an existing feature with overwrite
    model.addSpline("B", "b", 0.0f, 1.0f, true)
    // add a new family with overwrite
    model.addSpline("C", "c", 0.0f, 1.0f, true)
    // add an existing
    val weights = model.getWeightSpline.asScala
    for (familyMap <- weights) {
      for (featureMap <- familyMap._2.asScala) {
        val family = familyMap._1
        val feature = featureMap._1
        val spline = featureMap._2.spline
        log.info(("family=%s,feature=%s,minVal=%f, maxVal=%f, weights=%s")
                   .format(family, feature, spline.getMinVal, spline.getMaxVal, spline.toString))
        if (family.equals("A")) {
          assertTrue(feature.equals("a"))
          assertEquals(spline.getMaxVal, 10.0f, 0.01f)
          assertEquals(spline.getMinVal, 1.0f, 0.01f)
          assertEquals(spline.evaluate(1.0f), 1.0f, 0.01f)
          assertEquals(spline.evaluate(10.0f), 2.0f, 0.01f)
        } else if (family.equals("B")) {
          assertTrue(feature.equals("b"))
          assertEquals(spline.getMaxVal, 1.0f, 0.01f)
          assertEquals(spline.getMinVal, 0.0f, 0.01f)
        } else {
          assertTrue(family.equals("C"))
          assertTrue(feature.equals("c"))
          assertEquals(spline.getMaxVal, 1.0f, 0.01f)
          assertEquals(spline.getMinVal, 0.0f, 0.01f)
        }
      }
    }
  }

  @Test
  def testSplineRegression(): Unit = {
    val (trainingExample, trainingLabel) = TrainingTestHelper.makeRegressionExamples()
    var sc = new SparkContext("local", "SplineRegressionTest")
    try {
      val config = ConfigFactory.parseString(makeRegressionConfig)
      val input = sc.parallelize(trainingExample)
      val model = SplineTrainer.train(sc, input, config, "model_config")
      TrainingTestHelper.printSpline(model)
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
      log.info("Testing: Average absolute error = %f".format(testError))
      assertTrue(testError < 3.0)
    } finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }
}
