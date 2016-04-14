package com.airbnb.aerosolve.training

import java.io.{StringReader, BufferedWriter, BufferedReader, StringWriter}

import com.airbnb.aerosolve.core.features.FeatureRegistry
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
      |  max_depth : 8
      |  min_leaf_items : 5
      |  num_tries : 100
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

  @Test
  def testForestTrainerMulticlassHellinger() = {
    val config = ConfigFactory.parseString(makeConfig("multiclass_hellinger"))
    ForestTrainerTestHelper.testForestTrainerMulticlass(config, false, 0.8)
  }

  @Test
  def testForestTrainerMulticlassGini() = {
    val config = ConfigFactory.parseString(makeConfig("multiclass_gini"))
    ForestTrainerTestHelper.testForestTrainerMulticlass(config, false, 0.8)
  }

  @Test
  def testForestTrainerMulticlassHellingerNonlinear() = {
    val config = ConfigFactory.parseString(makeConfig("multiclass_hellinger"))
    ForestTrainerTestHelper.testForestTrainerMulticlassNonlinear(config, false, 0.7)
  }

  @Test
  def testForestTrainerMulticlassGiniNonlinear() = {
    val config = ConfigFactory.parseString(makeConfig("multiclass_gini"))
    ForestTrainerTestHelper.testForestTrainerMulticlassNonlinear(config, false, 0.7)
  }

}

object ForestTrainerTestHelper {
  val log = LoggerFactory.getLogger("ForestTrainerTest")
  val registry = new FeatureRegistry

  def testForestTrainer(config : Config, boost : Boolean, expectedCorrect : Double) = {
    testForestTrainerHelper(config, boost, expectedCorrect, false, false)
  }

  def testForestTrainerMulticlass(config : Config, boost : Boolean, expectedCorrect : Double) = {
    testForestTrainerHelper(config, boost, expectedCorrect, true, false)
  }

  def testForestTrainerMulticlassNonlinear(config : Config, boost : Boolean, expectedCorrect : Double) = {
    testForestTrainerHelper(config, boost, expectedCorrect, true, true)
  }

  def testForestTrainerHelper(config : Config,
                              boost : Boolean,
                              expectedCorrect : Double,
                              multiclass : Boolean,
                              nonlinear : Boolean) = {

    var examples = ArrayBuffer[Example]()
    var label = ArrayBuffer[Double]()
    var labels = ArrayBuffer[String]()
    var numPos = 0

    if (multiclass) {
      val (tmpEx, tmpLabels) = if (nonlinear)
        TrainingTestHelper.makeNonlinearMulticlassClassificationExamples(registry) else
        TrainingTestHelper.makeSimpleMulticlassClassificationExamples(false, registry)
      examples = tmpEx
      labels = tmpLabels
    } else {
      val (tmpEx, tmpLabel, tmpNumPos) = TrainingTestHelper.makeClassificationExamples(registry)
      examples = tmpEx
      label = tmpLabel
      numPos = tmpNumPos
    }

    var sc = new SparkContext("local", "ForestTrainerTest")

    try {
      val input = sc.parallelize(examples)
      val model = if (boost) {
        BoostedForestTrainer.train(sc, input, config, "model_config", registry)
      } else {
        ForestTrainer.train(sc, input, config, "model_config", registry)
      }

      val trees = model.trees.asScala
      for (tree <- trees) {
        log.info("Tree:")
        val stumps = tree.stumps.asScala
        stumps.foreach(stump => log.info(stump.toString))
      }

      if (multiclass) {
        var numCorrect: Int = 0
        for (i <- 0 until examples.length) {
          val ex = examples(i)
          val scores = model.scoreItemMulticlass(ex.only)
          val best = scores.asScala.sortWith((a, b) => a.getScore > b.getScore).head
          if (best.getLabel == labels(i)) {
            numCorrect = numCorrect + 1
          }
        }
        val fracCorrect: Double = numCorrect * 1.0 / examples.length
        log.info("Num correct = %d, frac correct = %f"
                   .format(numCorrect, fracCorrect))
        assertTrue(fracCorrect > expectedCorrect)
      } else {
        var numCorrect : Int = 0;
        var i : Int = 0;
        val labelArr = label.toArray
        for (ex <- examples) {
          val score = model.scoreItem(ex.only)
          if (score * labelArr(i) > 0) {
            numCorrect += 1
          }
          i += 1
        }
        val fracCorrect : Double = numCorrect * 1.0 / examples.length
        log.info("Num correct = %d, frac correct = %f, num pos = %d, num neg = %d"
                   .format(numCorrect, fracCorrect, numPos, examples.length - numPos))
        assertTrue(fracCorrect > expectedCorrect)
      }

      val swriter = new StringWriter()
      val writer = new BufferedWriter(swriter)
      model.save(writer)
      writer.close()
      val str = swriter.toString
      val sreader = new StringReader(str)
      val reader = new BufferedReader(sreader)

      val model2Opt = ModelFactory.createFromReader(reader, registry)
      assertTrue(model2Opt.isPresent)
      val model2 = model2Opt.get()

      if (multiclass) {
        for (ex <- examples) {
          val score = model.scoreItemMulticlass(ex.only)
          val score2 = model2.scoreItemMulticlass(ex.only)
          assertEquals(score.size, score2.size)
          for (i <- 0 until score.size) {
            assertEquals(score.get(i).getScore, score2.get(i).getScore, 0.1d)
          }
        }
      } else {
        for (ex <- examples) {
          val score = model.scoreItem(ex.only)
          val score2 = model2.scoreItem(ex.only)
          assertEquals(score, score2, 0.01d)
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
