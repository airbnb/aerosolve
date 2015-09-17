package com.airbnb.aerosolve.training

import java.io.{StringReader, BufferedWriter, BufferedReader, StringWriter}

import com.airbnb.aerosolve.core.models.ModelFactory
import com.airbnb.aerosolve.core.{Example, FeatureVector}
import com.typesafe.config.Config
import com.typesafe.config.ConfigFactory
import org.apache.spark.SparkContext
import org.junit.Test
import org.slf4j.LoggerFactory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import scala.collection.JavaConverters._

import scala.collection.mutable.ArrayBuffer

class BoostedForestTrainerTest {
  val log = LoggerFactory.getLogger("BoostedForestTrainerTest")

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
      |  max_depth : 2
      |  min_leaf_items : 5
      |  num_tries : 10
      |  num_trees : 10
      |  iterations : 3
      |  learning_rate : 0.1
      |  subsample : 0.5
      |  context_transform : identity_transform
      |  item_transform : identity_transform
      |  combined_transform : identity_transform
      |}
    """.stripMargin
      .format(splitCriteria)
  }
  
  @Test
  def testBoostedForestTrainerHellinger() = {
    val config = ConfigFactory.parseString(makeConfig("hellinger"))
    ForestTrainerTestHelper.testForestTrainer(config, true, 0.8)
  }
  
  @Test
  def testBoostedForestTrainerGini() = {
    val config = ConfigFactory.parseString(makeConfig("gini"))
    ForestTrainerTestHelper.testForestTrainer(config, true, 0.8)
  }
  
  @Test
  def testBoostedForestTrainerInformationGain() = {
    val config = ConfigFactory.parseString(makeConfig("information_gain"))
    ForestTrainerTestHelper.testForestTrainer(config, true, 0.8)
  }  
}
