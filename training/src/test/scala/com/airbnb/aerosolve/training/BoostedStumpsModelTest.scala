package com.airbnb.aerosolve.training

import java.io.{StringReader, BufferedWriter, BufferedReader, StringWriter}

import com.airbnb.aerosolve.core.models.ModelFactory
import com.airbnb.aerosolve.core.{Example, FeatureVector}
import com.typesafe.config.ConfigFactory
import org.apache.spark.SparkContext
import org.junit.Test
import org.slf4j.LoggerFactory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import scala.collection.JavaConverters._

import scala.collection.mutable.ArrayBuffer

class BoostedStumpsModelTest {
  val log = LoggerFactory.getLogger("BoostedStumpsModelTest")

  def makeConfig(loss : String) : String = {
    """
      |identity_transform {
      |  transform : list
      |  transforms: []
      |}
      |model_config {
      |  loss : "%s"
      |  num_bags : 1
      |  rank_key : "$rank"
      |  num_candidates : 100
      |  rank_threshold : 0.0
      |  dropout : 0.0
      |  learning_rate : 0.1
      |  lambda : 0.1
      |  lambda2 : 0.1
      |  subsample : 0.1
      |  iterations : 10
      |  context_transform : identity_transform
      |  item_transform : identity_transform
      |  combined_transform : identity_transform
      |}
    """.stripMargin.format(loss)
  }

  @Test
  def testBoostedStumpTrainerLogistic : Unit = {
    testBoostedStumpTrainer("logistic")
  }

  @Test
  def testBoostedStumpTrainerHinge : Unit = {
    testBoostedStumpTrainer("hinge")
  }

  def testBoostedStumpTrainer(loss : String) = {
    val (examples, label, numPos) = TrainingTestHelper.makeSimpleClassificationExamples

    var sc = new SparkContext("local", "BoostedStumpsTEst")

    try {
      val config = ConfigFactory.parseString(makeConfig(loss))

      val input = sc.parallelize(examples)
      val model = BoostedStumpsTrainer.train(sc, input, config, "model_config")

      val stumps = model.getStumps.asScala
      stumps.foreach(stump => log.info(stump.toString))

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
      assertTrue(fracCorrect > 0.6)

      val swriter = new StringWriter()
      val writer = new BufferedWriter(swriter)
      model.save(writer)
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
