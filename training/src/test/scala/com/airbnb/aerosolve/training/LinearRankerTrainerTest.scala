package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core.features.{SimpleExample, FeatureRegistry}
import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.core.{Example, FeatureVector}
import com.typesafe.config.{ConfigFactory, Config}
import org.apache.spark.SparkContext
import org.junit.Test
import org.slf4j.LoggerFactory
import org.junit.Assert.assertTrue

import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._
import scala.collection.JavaConversions._

class LinearRankerTrainerTest {
  val log = LoggerFactory.getLogger("LinearRankerTrainerTest")
  val registry = new FeatureRegistry


  // Creates an example with the context being a name
  // and items being integers from 0 to 10 and the user "name"
  // likes integers mod like_mod == 0
  def makeExamples(examples : ArrayBuffer[Example],
                   name : String, likes_mod : Int) = {
    for (i <- 0 to 10) {
      for (j <- 1 to 10) {
        val example = new SimpleExample(registry)
        example.context.putString("name", name)
        addItem(example, likes_mod, i)
        addItem(example, likes_mod, j)
        examples += example
      }
    }
  }

  def addItem(example : Example, likes_mod : Int, i : Int) = {
    val rank : Double = if (i % likes_mod == 0) {
      1.0
    } else {
      0.0
    }
    val item: FeatureVector = example.createVector()
    item.putString("number", s"$i")
    item.put("$rank", "", rank)
  }

  def makeConfig: String = {
    """
      |cross_transform {
      |  transform : cross
      |  field1: name
      |  field2: number
      |  output: name_X_number
      |}
      |identity_transform {
      |  transform : list
      |  transforms: []
      |}
      |model_config {
      |  num_bags : 1
      |  rank_key : "$rank"
      |  learning_rate : 1.0
      |  lambda : 0.4
      |  lambda2 : 0.1
      |  iterations : 10
      |  dropout : 0.1
      |  context_transform : identity_transform
      |  item_transform : identity_transform
      |  combined_transform : cross_transform
      |}
    """.stripMargin
  }
  @Test def testLinearRankerTrainer {
    val examples = ArrayBuffer[Example]()
    makeExamples(examples, "alice", 2)
    makeExamples(examples, "bob", 3)
    makeExamples(examples, "charlie", 4)

    var sc = new SparkContext("local", "LinearRankerTrainerTest")

    try {

      val config = ConfigFactory.parseString(makeConfig)

      val input = sc.parallelize(examples)
      val origWeights = LinearRankerTrainer.train(sc, input, config, "model_config", registry)
      val weights = origWeights.toMap

      origWeights
        .foreach(wt => {
        log.info("%s:%s=%f".format(wt._1.family.name, wt._1.name, wt._2))
      })

      // Ensure alice likes even numbers
      assertTrue(weights.getOrElse(registry.feature("name_X_number", "alice^2"), 0.0) >
                 weights.getOrElse(registry.feature("name_X_number", "alice^5"), 0.0))

      // Ensure bob likes multiples of 3
      assertTrue(weights.getOrElse(registry.feature("name_X_number", "bob^6"), 0.0) >
                 weights.getOrElse(registry.feature("name_X_number", "bob^1"), 0.0))

      // Ensure charlie likes multiples of 4
      assertTrue(weights.getOrElse(registry.feature("name_X_number", "charlie^8"), 0.0) >
                 weights.getOrElse(registry.feature("name_X_number", "charlie^7"), 0.0))
    } finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }
  @Test def TransformExamplesTest {
    val examples = ArrayBuffer[Example]()
    makeExamples(examples, "alice", 2)

    var sc = new SparkContext("local", "TransformExamplesTest")

    try {

      val config = ConfigFactory.parseString(makeConfig)

      val input = sc.parallelize(examples)
      val xform = LinearRankerUtils
      .transformExamples(input, config, "model_config", registry)
      .collect
      .toArray
      .head
      log.info(xform.toString)
      val fv = xform.asScala.toArray
      val expected = List((0, "name", "alice"),
                          (0, "number", "0"),
                          (0, "name_X_number", "alice^0"),
                          (1, "name", "alice"),
                          (1, "number", "1"),
                          (1, "name_X_number", "alice^1"))
      expected.foreach{ case (index, familyName, expectedValue) =>
        assertTrue(fv(index).get(registry.family(familyName)).iterator.next
                     .feature.name.equals(expectedValue))
      }
    } finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }
  
}