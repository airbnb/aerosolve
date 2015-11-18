package com.airbnb.aerosolve.training

import java.util

import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.core.{Example, FeatureVector}
import com.typesafe.config.{ConfigFactory, Config}
import org.apache.spark.SparkContext
import org.junit.Test
import org.slf4j.LoggerFactory
import org.junit.Assert.assertEquals

import scala.collection.mutable.ArrayBuffer

class LinearRegressionTrainerTest {
  val log = LoggerFactory.getLogger("LinearRegressionTrainerTest")

  // Creates an example with name and target.
  def makeExamples(examples : ArrayBuffer[Example],
                   name : String,
                   target : Double) = {
    val example = new Example
    val item: FeatureVector = new FeatureVector
    item.setStringFeatures(new java.util.HashMap)
    val itemSet = new java.util.HashSet[String]()
    itemSet.add(name)
    val stringFeatures = item.getStringFeatures
    stringFeatures.put("name", itemSet)
    val biasSet = new java.util.HashSet[String]()
    biasSet.add("1")
    stringFeatures.put("bias", biasSet)
    item.setFloatFeatures(new java.util.HashMap)
    val floatFeatures = item.getFloatFeatures
    floatFeatures.put("$rank", new java.util.HashMap)
    floatFeatures.get("$rank").put("", target)
    example.addToExample(item)
    examples += example
  }

  def makeConfig(loss : String): String = {
    """
      |identity_transform {
      |  transform : list
      |  transforms: []
      |}
      |model_config {
      |  num_bags : 1
      |  loss : "%s"
      |  rank_key : "$rank"
      |  lambda : 0.01
      |  lambda2 : 0.01
      |  epsilon : 0.1
      |  learning_rate : 1.0
      |  iterations : 10
      |  context_transform : identity_transform
      |  item_transform : identity_transform
      |  combined_transform : identity_transform
      |}
    """.stripMargin.format(loss)
  }

  @Test
  def testLinearRegressionTrainer() {
    testLinearRegressionTrainerWithLoss("regression")
  }
  
  @Test
  def testLinearRegressionL2Trainer() {
    testLinearRegressionTrainerWithLoss("regressionL2")
  }
  
  def testLinearRegressionTrainerWithLoss(loss: String) {
    val examples = ArrayBuffer[Example]()
    for (i <- 0 until 10) {
      makeExamples(examples, "alice", 2 + 0.1 * scala.util.Random.nextDouble())
      makeExamples(examples, "bob", 3 + 0.1 * scala.util.Random.nextDouble())
      makeExamples(examples, "charlie", 7 + 0.1 * scala.util.Random.nextDouble())
    }

    var sc = new SparkContext("local", "RegressionTrainerTest")

    try {
      val config = ConfigFactory.parseString(makeConfig(loss))

      val input = sc.parallelize(examples)
      val origWeights = LinearRankerTrainer.train(sc, input, config, "model_config")
      val weights = origWeights.toMap

      origWeights
        .foreach(wt => {
        log.info("%s:%s=%f".format(wt._1._1, wt._1._2, wt._2))
      })

      // Ensure alice likes 2
      assertEquals(2.0,
                   weights.getOrElse(("name", "alice"), 0.0) +
                   weights.getOrElse(("bias", "1"), 0.0),
                   0.5)

      // Ensure bob likes 3
      assertEquals(3.0,
                   weights.getOrElse(("name", "bob"), 0.0) +
                   weights.getOrElse(("bias", "1"), 0.0),
                   0.5)

      // Ensure charlie 7
      assertEquals(7.0,
                   weights.getOrElse(("name", "charlie"), 0.0) +
                   weights.getOrElse(("bias", "1"), 0.0),
                   0.5)

    } finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }
}