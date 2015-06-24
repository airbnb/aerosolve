package com.airbnb.aerosolve.training

import java.util

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
      val origWeights = LinearRankerTrainer.train(sc, input, config, "model_config")
      val weights = origWeights.toMap

      origWeights
        .foreach(wt => {
        log.info("%s:%s=%f".format(wt._1._1, wt._1._2, wt._2))
      })

      for (j <- 0 until 10) {
        val name = j.toString
        if (j % 2 == 0) {
          assertTrue(weights.getOrElse(("name", name), 0.0) >= 1.0)
        } else {
          assertTrue(weights.getOrElse(("name", name), 0.0) <= -1.0)
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