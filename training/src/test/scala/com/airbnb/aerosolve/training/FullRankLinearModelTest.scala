package com.airbnb.aerosolve.training

import java.io.{StringReader, BufferedWriter, BufferedReader, StringWriter}

import com.airbnb.aerosolve.core.models.ModelFactory
import com.typesafe.config.ConfigFactory
import org.apache.spark.SparkContext
import org.junit.Test
import org.slf4j.LoggerFactory
import org.junit.Assert._

import scala.collection.JavaConverters._

class FullRankLinearModelTest {
  val log = LoggerFactory.getLogger("FullRankLinearModelTest")

  def makeConfig(loss : String, lambda : Double, solver : String) : String = {
    """
      |identity_transform {
      |  transform : list
      |  transforms: []
      |}
      |model_config {
      |  rank_key : "$rank"
      |  loss : "%s"
      |  subsample : 0.5
      |  iterations : 10
      |  lambda : %f
      |  min_count : 0
      |  cache : "memory"
      |  solver : "%s"
      |  context_transform : identity_transform
      |  item_transform : identity_transform
      |  combined_transform : identity_transform
      |}
    """.stripMargin
       .format(loss, lambda, solver)
  }

  @Test
  def testFullRankLinearSoftmax() = {
    testFullRankLinear("softmax", 0.1, "sparse_boost", false, 0.9)
  }

  @Test
  def testFullRankLinearHinge() = {
    testFullRankLinear("hinge", 0.1, "sparse_boost", false, 0.85)
  }

  @Test
  def testFullRankLinearHingeRprop() = {
    testFullRankLinear("hinge", 0.1, "rprop", false, 0.9)
  }

  @Test
  def testFullRankLinearHingeMultilabel() = {
    testFullRankLinear("hinge", 0.1, "rprop", true, 0.9)
  }

  @Test
  def testFullRankLinearSquaredHinge() = {
    testFullRankLinear("squared_hinge", 0.1, "sparse_boost", false, 0.9)
  }

  @Test
  def testFullRankLinearSquaredHingeMultilabel() = {
    testFullRankLinear("squared_hinge", 0.1, "sparse_boost", true, 0.9)
  }

  @Test
  def testFullRankLinearSquaredHingeRprop() = {
    testFullRankLinear("squared_hinge", 0.1, "rprop", false, 0.9)
  }

  @Test
  def testFullRankLinearSquaredHingeMultilabelRprop() = {
    testFullRankLinear("squared_hinge", 0.1, "rprop", true, 0.9)
  }

  def testFullRankLinear(loss : String,
                         lambda : Double,
                         solver : String,
                         multiLabel : Boolean,
                         expectedCorrect : Double) = {
    val (examples, labels) = TrainingTestHelper.makeSimpleMulticlassClassificationExamples(multiLabel)

    var sc = new SparkContext("local", "FullRankLinearTest")

    try {
      val config = ConfigFactory.parseString(makeConfig(loss, lambda, solver))

      val input = sc.parallelize(examples)
      val model = FullRankLinearTrainer.train(sc, input, config, "model_config")

      val weightVector = model.getWeightVector.asScala
      for (wv <- weightVector) {
        log.info(wv.toString())
      }

      var numCorrect: Int = 0
      for (i <- 0 until examples.length) {
        val ex = examples(i)
        val scores = model.scoreItemMulticlass(ex.example.get(0))
        val best = scores.asScala.sortWith((a, b) => a.score > b.score).head
        if (best.label == labels(i)) {
          numCorrect = numCorrect + 1
        }
      }
      val fracCorrect: Double = numCorrect * 1.0 / examples.length
      log.info("Num correct = %d, frac correct = %f"
                 .format(numCorrect, fracCorrect))
      assertTrue(fracCorrect > expectedCorrect)

      val swriter = new StringWriter()
      val writer = new BufferedWriter(swriter)
      model.save(writer)
      writer.close()
      val str = swriter.toString
      val sreader = new StringReader(str)
      val reader = new BufferedReader(sreader)

      val model2Opt = ModelFactory.createFromReader(reader)
      assertTrue(model2Opt.isPresent)
      val model2 = model2Opt.get()
      val labelCount = if (multiLabel) 6 else 4

      for (ex <- examples) {
        val score = model.scoreItemMulticlass(ex.example.get(0))
        val score2 = model2.scoreItemMulticlass(ex.example.get(0))
        assertEquals(score.size, labelCount)
        assertEquals(score.size, score2.size)
        for (i <- 0 until labelCount) {
          assertEquals(score.get(i).score, score2.get(i).score, 0.1f)
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
