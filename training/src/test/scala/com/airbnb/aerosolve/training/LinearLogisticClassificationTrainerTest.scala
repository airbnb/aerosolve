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
import org.junit.Assert.assertTrue

import scala.collection.mutable.ArrayBuffer

class LinearLogisticClassificationTrainerTest {
  val log = LoggerFactory.getLogger("LinearLogisticClassificationTrainerTest")
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

  def makeConfig: String = {
    """
      |identity_transform {
      |  transform : list
      |  transforms: []
      |}
      |model_config {
      |  num_bags : 1
      |  loss : "logistic"
      |  rank_key : "$rank"
      |  rank_threshold : 0.0
      |  learning_rate : 1.0
      |  iterations : 10
      |  lambda : 0.1
      |  lambda2 : 0.01
      |  dropout : 0.1
      |  context_transform : identity_transform
      |  item_transform : identity_transform
      |  combined_transform : identity_transform
      |}
    """.stripMargin
  }
  @Test def testLinearClassificationTrainer {
    val examples = ArrayBuffer[Example]()
    for (i <- 0 until 10) {
      for (j <- 0 until 10) {
        val name = j.toString
        val rank = if (j % 2 == 0) {
          1.0
        } else {
          -1.0
        }
        makeExamples(examples, name, rank)
      }
    }

    var sc = new SparkContext("local", "LogisticClassificationTrainerTest")

    try {
      val config = ConfigFactory.parseString(makeConfig)

      val input = sc.parallelize(examples)
      val origWeights = LinearRankerTrainer.train(sc, input, config, "model_config", registry)
      val weights = origWeights.toMap

      origWeights
        .foreach(wt => {
        log.info("%s:%s=%f".format(wt._1.family.name, wt._1.name, wt._2))
      })

      for (j <- 0 until 10) {
        val name = j.toString
        if (j % 2 == 0) {
          assertTrue(weights.getOrElse(registry.feature("name", name), 0.0) >= 1.0)
        } else {
          assertTrue(weights.getOrElse(registry.feature("name", name), 0.0) <= -1.0)
        }
      }
   } finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }
}