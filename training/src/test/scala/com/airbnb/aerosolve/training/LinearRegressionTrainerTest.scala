package com.airbnb.aerosolve.training

import java.util

import com.airbnb.aerosolve.core.features.{SimpleExample, FeatureRegistry}
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
  val registry = new FeatureRegistry
  // Creates an example with name and target.
  def makeExamples(examples : ArrayBuffer[Example],
                   name : String,
                   target : Double) = {
    val example = new SimpleExample(registry)
    val item: FeatureVector = example.createVector()
    item.putString("name", name)
    item.putString("bias", "1")
    item.put("$rank", "", target)
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
      val origWeights = LinearRankerTrainer.train(sc, input, config, "model_config", registry)
      val weights = origWeights.toMap

      origWeights
        .foreach(wt => {
        log.info("%s:%s=%f".format(wt._1.family.name, wt._1.name, wt._2))
      })

      // Ensure alice likes 2
      assertEquals(2.0,
                   weights.getOrElse(registry.feature("name", "alice"), 0.0) +
                   weights.getOrElse(registry.feature("bias", "1"), 0.0),
                   0.5)

      // Ensure bob likes 3
      assertEquals(3.0,
                   weights.getOrElse(registry.feature("name", "bob"), 0.0) +
                   weights.getOrElse(registry.feature("bias", "1"), 0.0),
                   0.5)

      // Ensure charlie 7
      assertEquals(7.0,
                   weights.getOrElse(registry.feature("name", "charlie"), 0.0) +
                   weights.getOrElse(registry.feature("bias", "1"), 0.0),
                   0.5)

    } finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }
}