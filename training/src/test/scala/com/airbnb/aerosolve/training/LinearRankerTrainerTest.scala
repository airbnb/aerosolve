package com.airbnb.aerosolve.training

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

  // Creates an example with the context being a name
  // and items being integers from 0 to 10 and the user "name"
  // likes integers mod like_mod == 0
  def makeExamples(examples : ArrayBuffer[Example],
                   name : String, likes_mod : Int) = {

    val context: FeatureVector = new FeatureVector
    context.setStringFeatures(new java.util.HashMap)
    val nameSet = new java.util.HashSet[String]()
    nameSet.add(name)
    val stringFeatures = context.getStringFeatures
    stringFeatures.put("name", nameSet)

    for (i <- 0 to 10) {
      for (j <- 1 to 10) {
        val example = new Example
        example.setContext(context)
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
    val item: FeatureVector = new FeatureVector
    item.setStringFeatures(new java.util.HashMap)
    val itemSet = new java.util.HashSet[String]()
    itemSet.add("%d".format(i))
    val stringFeatures = item.getStringFeatures
    stringFeatures.put("number", itemSet)
    item.setFloatFeatures(new java.util.HashMap)
    val floatFeatures = item.getFloatFeatures
    floatFeatures.put("$rank", new java.util.HashMap)
    floatFeatures.get("$rank").put("", rank)
    example.addToExample(item)
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
      val origWeights = LinearRankerTrainer.train(sc, input, config, "model_config")
      val weights = origWeights.toMap

      origWeights
        .foreach(wt => {
        log.info("%s:%s=%f".format(wt._1._1, wt._1._2, wt._2))
      })

      // Ensure alice likes even numbers
      assertTrue(weights.getOrElse(("name_X_number", "alice^2"), 0.0) >
                 weights.getOrElse(("name_X_number", "alice^5"), 0.0))

      // Ensure bob likes multiples of 3
      assertTrue(weights.getOrElse(("name_X_number", "bob^6"), 0.0) >
                 weights.getOrElse(("name_X_number", "bob^1"), 0.0))

      // Ensure charlie likes multiples of 4
      assertTrue(weights.getOrElse(("name_X_number", "charlie^8"), 0.0) >
                 weights.getOrElse(("name_X_number", "charlie^7"), 0.0))
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
      .transformExamples(input, config, "model_config")
      .collect
      .toArray
      .head
      log.info(xform.toString)
      val fv = xform.example.asScala.toArray
      assertTrue(fv(0).stringFeatures.get("name").asScala.head.equals("alice"))
      assertTrue(fv(0).stringFeatures.get("number").asScala.head.equals("0"))
      assertTrue(fv(0).stringFeatures.get("name_X_number").asScala.head.equals("alice^0"))
      assertTrue(fv(1).stringFeatures.get("name").asScala.head.equals("alice"))
      assertTrue(fv(1).stringFeatures.get("number").asScala.head.equals("1"))
      assertTrue(fv(1).stringFeatures.get("name_X_number").asScala.head.equals("alice^1"))
    } finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }
  
}