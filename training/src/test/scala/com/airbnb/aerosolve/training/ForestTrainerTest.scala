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

class ForestTrainerTest {
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
      |  max_depth : 5
      |  min_leaf_items : 5
      |  num_tries : 10
      |  num_trees : 5
      |  context_transform : identity_transform
      |  item_transform : identity_transform
      |  combined_transform : identity_transform
      |}
    """.stripMargin
      .format(splitCriteria)
  }

  @Test
  def testForestTrainerHellinger() = {
    val config = ConfigFactory.parseString(makeConfig("hellinger"))
    ForestTrainerTestHelper.testForestTrainer(config, false, 0.8)
  }
  
  @Test
  def testForestTrainerGini() = {
    val config = ConfigFactory.parseString(makeConfig("gini"))
    ForestTrainerTestHelper.testForestTrainer(config, false, 0.8)
  }
  
  @Test
  def testForestTrainerInformationGain() = {
    val config = ConfigFactory.parseString(makeConfig("information_gain"))
    ForestTrainerTestHelper.testForestTrainer(config, false, 0.8)
  }
  
}

object ForestTrainerTestHelper {
  val log = LoggerFactory.getLogger("ForestTrainerTest")

  def makeExample(x : Double,
                  y : Double,
                  target : Double) : Example = {
    val example = new Example
    val item: FeatureVector = new FeatureVector
    item.setFloatFeatures(new java.util.HashMap)
    val floatFeatures = item.getFloatFeatures
    floatFeatures.put("$rank", new java.util.HashMap)
    floatFeatures.get("$rank").put("", target)
    floatFeatures.put("loc", new java.util.HashMap)
    val loc = floatFeatures.get("loc")
    loc.put("x", x)
    loc.put("y", y)
    example.addToExample(item)
    return example
  }

  def testForestTrainer(config : Config, boost : Boolean, expectedCorrect : Double) = {
    val examples = ArrayBuffer[Example]()
    val label = ArrayBuffer[Double]()
    val rnd = new java.util.Random(1234)
    var numPos : Int = 0;
    for (i <- 0 until 200) {
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

    var sc = new SparkContext("local", "ForestTrainerTest")

    try {
      val input = sc.parallelize(examples)
      val model = if (boost) {
        BoostedForestTrainer.train(sc, input, config, "model_config")
      } else {
        ForestTrainer.train(sc, input, config, "model_config")
      }

      val trees = model.getTrees.asScala
      for (tree <- trees) {
        log.info("Tree:")
        val stumps = tree.getStumps.asScala
        stumps.foreach(stump => log.info(stump.toString))
      }

      var numCorrect : Int = 0;
      var i : Int = 0;
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

      val swriter = new StringWriter();
      val writer = new BufferedWriter(swriter);
      model.save(writer);
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
}
