package com.airbnb.aerosolve.training

import java.io.{BufferedReader, BufferedWriter, StringReader, StringWriter}
import java.util

import com.airbnb.aerosolve.core.features.{SimpleExample, FeatureRegistry}
import com.airbnb.aerosolve.core.models.SplineModel.WeightSpline
import com.airbnb.aerosolve.core.models.{ModelFactory, SplineModel}
import com.airbnb.aerosolve.core.{Example, FeatureVector}
import com.airbnb.aerosolve.core.transforms.Transformer
import java.util.{HashMap, Scanner}

import com.airbnb.aerosolve.core.functions.Spline
import com.typesafe.config.ConfigFactory
import org.apache.spark.SparkContext
import org.junit.Test
import org.slf4j.LoggerFactory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

class SplineRankingTrainerTest {
  val log = LoggerFactory.getLogger("SplineTrainerTest")
  val registry = new FeatureRegistry

  // Creates an example with the context being a name
  // and items being integers from 0 to 10 and the user "name"
  // likes integers mod like_mod == 0
  def makeExample(name : String, likes_mod : Int): Example = {
    val example = new SimpleExample(registry)
    example.context.putString("name", name)

    for (i <- 0 to 10) {
      addItem(example, likes_mod, i)
    }
    example
  }

  def addItem(example : Example, likes_mod : Int, i : Int) = {
    val rank : Double = if (i % likes_mod == 0) {
      1.0
    } else {
      -1.0
    }
    val item: FeatureVector = example.createVector()
    item.put("number","n", i.toDouble)
    item.put("$rank", "", rank)
  }

  def makeConfig() : String = {
    """
      |identity_transform {
      |  transform : list
      |  transforms: []
      |}
      |name_and_number {
      |  transform : string_cross_float
      |  field1 : "name"
      |  field2 : "number"
      |  output : "name_X_num"
      |}
      |model_config {
      |  num_bags : 1
      |  loss : "rank_and_hinge"
      |  rank_fraction : 0.5
      |  rank_margin : 1.0
      |  max_samples_per_example : 10
      |  margin : 1.0
      |  rank_key : "$rank"
      |  rank_threshold : 0.0
      |  learning_rate : 0.5
      |  num_bins : 16
      |  iterations : 20
      |  smoothing_tolerance : 0.1
      |  linfinity_threshold : 0.01
      |  linfinity_cap : 1.0
      |  dropout : 0.1
      |  min_count : 0
      |  subsample : 1.0
      |  context_transform : identity_transform
      |  item_transform : identity_transform
      |  combined_transform : name_and_number
      |}
    """.stripMargin
  }

  @Test
  def testSplineRankingTrainer() = {
    val alice = makeExample("alice", 2)
    val bob = makeExample("bob", 4)
    
    val examples = Array(alice, bob)

    var sc = new SparkContext("local", "SplineRankingTest")

    try {
      val config = ConfigFactory.parseString(makeConfig())

      val input = sc.parallelize(examples)
      val model = SplineTrainer.train(sc, input, config, "model_config", registry)

      val transformer = new Transformer(config, "model_config", registry)
      alice.transform(transformer)
      val aliceEx = alice.asScala.toSeq
      for (i <- 0 until 10) {
        log.info(model.scoreItem(aliceEx(i)).toString)
        log.info(aliceEx(i).toString)
      }
      // Alice likes even numbers
      assertTrue(model.scoreItem(aliceEx(2)) > model.scoreItem(aliceEx(1)))
      assertTrue(model.scoreItem(aliceEx(4)) > model.scoreItem(aliceEx(3)))
      assertTrue(model.scoreItem(aliceEx(6)) > model.scoreItem(aliceEx(9)))

      bob.transform(transformer)
      val bobEx = bob.asScala.toSeq
      for (i <- 0 until 10) {
        log.info(model.scoreItem(bobEx(i)).toString)
        log.info(bobEx(i).toString)
      }
      // Bob likes multiples of 4
      assertTrue(model.scoreItem(bobEx(4)) > model.scoreItem(bobEx(2)))
      assertTrue(model.scoreItem(bobEx(8)) > model.scoreItem(bobEx(6)))
      assertTrue(model.scoreItem(bobEx(0)) > model.scoreItem(bobEx(5)))
      assertTrue(model.scoreItem(bobEx(4)) > model.scoreItem(bobEx(1)))
    } finally {
      sc.stop
      sc = null
      // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }

}
