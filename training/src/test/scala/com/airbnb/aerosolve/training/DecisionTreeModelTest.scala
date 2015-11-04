package com.airbnb.aerosolve.training

import java.io.{StringReader, BufferedWriter, BufferedReader, StringWriter}

import com.airbnb.aerosolve.core.models.ModelFactory
import com.airbnb.aerosolve.core.ModelRecord
import com.typesafe.config.ConfigFactory
import org.apache.spark.SparkContext
import org.junit.Test
import org.slf4j.LoggerFactory
import org.junit.Assert._

import scala.collection.JavaConverters._
import scala.collection.JavaConversions

class DecisionTreeModelTest {
  val log = LoggerFactory.getLogger("DecisionTreeModelTest")

  def makeConfig(splitCriteria : String) : String = {
    """
      |identity_transform {
      |  transform : list
      |  transforms: []
      |}
      |model_config {
      |  rank_key : "$rank"
      |  split_criteria : "%s"
      |  num_candidates : 1000
      |  rank_threshold : 0.0
      |  max_depth : 20
      |  min_leaf_items : 5
      |  num_tries : 10
      |  context_transform : identity_transform
      |  item_transform : identity_transform
      |  combined_transform : identity_transform
      |}
    """.stripMargin
       .format(splitCriteria)
  }

  @Test
  def testDecisionTreeTrainerHellinger() = {
    testDecisionTreeClassificationTrainer("hellinger", 0.8)
  }

  @Test
  def testDecisionTreeTrainerGini() = {
    testDecisionTreeClassificationTrainer("gini", 0.8)
  }

  @Test
  def testDecisionTreeTrainerInformationGain() = {
    testDecisionTreeClassificationTrainer("information_gain", 0.8)
  }

  @Test
  def testDecisionTreeTrainerVariance() = {
    testDecisionTreeRegressionTrainer("variance")
  }

  def testDecisionTreeClassificationTrainer(splitCriteria : String,
                                            expectedCorrect : Double) = {
    val (examples, label, numPos) = TrainingTestHelper.makeClassificationExamples

    var sc = new SparkContext("local", "DecisionTreeModelTest")

    try {
      val config = ConfigFactory.parseString(makeConfig(splitCriteria))

      val input = sc.parallelize(examples)
      val model = DecisionTreeTrainer.train(sc, input, config, "model_config")

      val stumps = model.getStumps.asScala
      stumps.foreach(stump => log.info(stump.toString))

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
      assertTrue(fracCorrect > expectedCorrect)

      val swriter = new StringWriter()
      val writer = new BufferedWriter(swriter)
      model.save(writer)
      writer.close()
      val str = swriter.toString()
      val sreader = new StringReader(str)
      val reader = new BufferedReader(sreader)

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

  def testDecisionTreeRegressionTrainer(splitCriteria : String) = {
    val (examples, label) = TrainingTestHelper.makeRegressionExamples()

    var sc = new SparkContext("local", "DecisionTreeModelTest")

    try {
      val config = ConfigFactory.parseString(makeConfig(splitCriteria))

      val input = sc.parallelize(examples)
      val model = DecisionTreeTrainer.train(sc, input, config, "model_config")

      val stumps = model.getStumps.asScala
      stumps.foreach(stump => log.info(stump.toString))

      val labelArr = label.toArray
      var i : Int = 0
      var totalError : Double = 0

      for (ex <- examples) {
        val score = model.scoreItem(ex.example.get(0))
        val exampleLabel = labelArr(i)

        totalError += math.abs(score - exampleLabel)

        i += 1
      }
      log.info("Average absolute error = %f".format(totalError / examples.size.toDouble))
      // Total error not too high
      assertTrue(totalError / examples.size.toDouble < 3.0)

      // Points in flat region result in score of min value (-8.0)
      val flatRegionExamples = List(
        TrainingTestHelper.makeExample(0, -3.5, 0),
        TrainingTestHelper.makeExample(0, 3.2, 0)
      )

      flatRegionExamples.foreach { flatRegionExample =>
        val score = model.scoreItem(flatRegionExample.example.get(0))

        assertEquals(score, -8.0, 1.0f)
      }
    } finally {
      sc.stop
      sc = null

      // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }

  @Test
  def testEvaluateRegressionSplit() = {
    val examples = Array(
      createJavaExample(1.1, 5.0),
      createJavaExample(1.2, 5.6),
      createJavaExample(1.25, 11.9),
      createJavaExample(1.5, 10.2),
      createJavaExample(1.8, 12.5),
      createJavaExample(2.5, 8.3),
      createJavaExample(2.9, 18.4)
    )
    val testSplit = new ModelRecord()

    testSplit.setFeatureFamily("loc")
    testSplit.setFeatureName("x")
    testSplit.setThreshold(1.3)

    val result = DecisionTreeTrainer.evaluateRegressionSplit(examples, "$rank", 1, "variance", Some(testSplit))

    // Verify that Welford's Method is consistent with standard, two-pass calculation
    val leftMean = (5.0 + 5.6 + 11.9) / 3.0
    val leftSumSq = math.pow(5.0 - leftMean, 2) + math.pow(5.6 - leftMean, 2) + math.pow(11.9 - leftMean, 2)

    val rightMean = (10.2 + 12.5 + 8.3 + 18.4) / 4.0
    val rightSumSq =
        math.pow(10.2 - rightMean, 2) + math.pow(12.5 - rightMean, 2) +
        math.pow(8.3 - rightMean, 2) + math.pow(18.4 - rightMean, 2)

    assertEquals(result.get, -1.0 * (leftSumSq + rightSumSq), 0.000001f)
  }

  def createJavaExample(x : Double, rank : Double) = {
    JavaConversions.mapAsJavaMap(
      Map(
        "loc" -> JavaConversions.mapAsJavaMap(Map("x" -> x)),
        "$rank" -> JavaConversions.mapAsJavaMap(Map("" -> rank))
      )
    ).asInstanceOf[
        java.util.Map[java.lang.String, java.util.Map[java.lang.String, java.lang.Double]]]
  }

}
