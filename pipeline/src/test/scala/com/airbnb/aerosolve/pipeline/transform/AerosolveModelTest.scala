package com.airbnb.aerosolve.pipeline.transform

import java.nio.charset.StandardCharsets

import com.airbnb.aerosolve.core.{FeatureVector, Example}
import com.typesafe.config.ConfigFactory
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.{Test, AfterClass, BeforeClass}

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer

/**
 *
 */
class AerosolveModelTest {

  var sc: SQLContext = _

  @BeforeClass
  def setup() = {
    val sparkConf = new SparkConf()
      .set("spark.default.parallelism", "1")
      .set("spark.sql.shuffle.partitions", "1")
      .set("spark.ui.enabled", "false")
    val context = new SparkContext("local", "Model.Test", sparkConf)
    sc = new SQLContext(context)
  }

  @AfterClass def teardown(): Unit = {
    sc.sparkContext.stop()
  }

  val config : String = {
    """
      |identity_transform {
      |  transform : list
      |  transforms: []
      |}
      |model_config {
      |  num_bags : 3
      |  loss : "hinge"
      |  rank_key : "$rank"
      |  rank_threshold : 0.0
      |  learning_rate : 0.1
      |  num_bins : 16
      |  iterations : 10
      |  smoothing_tolerance : 0.1
      |  linfinity_threshold : 0.01
      |  linfinity_cap : 10.0
      |  dropout : 0.0
      |  min_count : 0
      |  subsample : 1.0
      |  context_transform : identity_transform
      |  item_transform : identity_transform
      |  combined_transform : identity_transform
      |  model_output : ""
      |}
    """.stripMargin
  }

  @Test
  def testAdditiveModel: Unit = {
    val (examples, label, numPos) = makeClassificationExamples
    val input = sc.sparkContext.parallelize(examples)
    //val model = AdditiveModelTrainer.train(sc, input, config, "model_config")
    //testClassificationModel(model, examples, label, numPos)
  }

  def makeClassificationExamples = {
    val examples = ArrayBuffer[Example]()
    val label = ArrayBuffer[Double]()
    val rnd = new java.util.Random(1234)
    var numPos : Int = 0
    for (i <- 0 until 500) {
      val x = 2.0 * rnd.nextDouble() - 1.0
      val y = 10.0 * (2.0 * rnd.nextDouble() - 1.0)
      val poly = x * x + 0.1 * y * y + 0.1 * x + 0.2 * y - 0.1 + Math.sin(x)
      val rank = if (poly < 1.0) {
        1.0
      } else {
        -1.0
      }
      if (rank > 0) numPos = numPos + 1
      label += rank
      examples += makeExample(x, y, rank)
    }
    (examples, label, numPos)
  }

  def makeExample(x : Double,
                  y : Double,
                  target : Double) : Example = {
    val example = new Example
    val item: FeatureVector = new FeatureVector
    item.setFloatFeatures(new java.util.HashMap)
    item.setStringFeatures(new java.util.HashMap)
    val floatFeatures = item.getFloatFeatures
    val stringFeatures = item.getStringFeatures
    // A string feature that is always on.
    stringFeatures.put("BIAS", new java.util.HashSet)
    stringFeatures.get("BIAS").add("B")
    // A string feature that is sometimes on
    if (x + y < 0) {
      stringFeatures.put("NEG", new java.util.HashSet)
      stringFeatures.get("NEG").add("T")
    }
    floatFeatures.put("$rank", new java.util.HashMap)
    floatFeatures.get("$rank").put("", target)
    floatFeatures.put("loc", new java.util.HashMap)
    val loc = floatFeatures.get("loc")
    loc.put("x", x)
    loc.put("y", y)
    example.addToExample(item)
    example
  }
}
