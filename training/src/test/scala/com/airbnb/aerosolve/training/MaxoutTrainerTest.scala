package com.airbnb.aerosolve.training

import java.io.{StringReader, BufferedWriter, BufferedReader, StringWriter}
import java.util

import com.airbnb.aerosolve.core.models.ModelFactory
import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.core.{Example, FeatureVector}
import com.typesafe.config.{ConfigFactory, Config}
import org.apache.spark.SparkContext
import org.junit.Test
import org.slf4j.LoggerFactory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import scala.collection.JavaConverters._

import scala.collection.mutable.ArrayBuffer

class MaxoutTrainerTest {
  val log = LoggerFactory.getLogger("MaxoutTrainerTest")

  def makeConfig(loss : String) : String = {
    """
      |identity_transform {
      |  transform : list
      |  transforms: []
      |}
      |model_config {
      |  num_bags : 1
      |  loss : "%s"
      |  rank_key : "$rank"
      |  rank_threshold : 0.0
      |  learning_rate : 0.5
      |  num_hidden : 16
      |  iterations : 10
      |  momentum : 0.9
      |  lambda : 0.01
      |  lambda2 : 0.01
      |  dropout_hidden : 0.5
      |  dropout : 0.0
      |  min_count : 0
      |  subsample : 1.0
      |  context_transform : identity_transform
      |  item_transform : identity_transform
      |  combined_transform : identity_transform
      |}
    """.stripMargin.format(loss)
  }

  @Test
  def testMaxoutTrainerLogistic : Unit = {
    testMaxoutTrainer("logistic")
  }

  @Test
  def testMaxoutTrainerHinge : Unit = {
    testMaxoutTrainer("hinge")
  }

  def testMaxoutTrainer(loss : String) = {
    val (examples, label, numPos) = TrainingTestHelper.makeClassificationExamples

    var sc = new SparkContext("local", "MaxoutTrainerText")

    try {
      val config = ConfigFactory.parseString(makeConfig(loss))

      val input = sc.parallelize(examples)
      val model = MaxoutTrainer.train(sc, input, config, "model_config")

      val weights = model.getWeightVector.asScala
      for (familyMap <- weights) {
        for (featureMap <- familyMap._2.asScala) {
          log.info(("family=%s,feature=%s,"
                    + "scale=%f, weights=%s")
                     .format(familyMap._1, featureMap._1, featureMap._2.scale,
                             featureMap._2.weights.toString()))
        }
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
      assertTrue(fracCorrect > 0.6)

      val swriter = new StringWriter();
      val writer = new BufferedWriter(swriter);
      model.save(writer);
      writer.close()
      val str = swriter.toString()
      val sreader = new StringReader(str)
      val reader = new BufferedReader(sreader)

      log.info(str)

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