package com.airbnb.aerosolve.training

import java.io.{StringReader, BufferedWriter, BufferedReader, StringWriter}

import com.airbnb.aerosolve.core.models.ModelFactory
import com.typesafe.config.ConfigFactory
import org.apache.spark.SparkContext
import org.junit.Test
import org.slf4j.LoggerFactory
import org.junit.Assert._

import scala.collection.JavaConverters._

class LowRankLinearTrainerTest {
  val log = LoggerFactory.getLogger("LowRankLinearTrainerTest")

  def makeConfig(lambda : Double, embeddingDim : Int, rankLossType : String) : String = {
    """
      |identity_transform {
      |  transform : list
      |  transforms: []
      |}
      |model_config {
      |  rank_key : "$rank"
      |  loss : "hinge"
      |  subsample : 1.0
      |  iterations : 10
      |  lambda : %f
      |  min_count : 0
      |  embedding_dimension : %d
      |  cache : "memory"
      |  rank_loss : "%s"
      |  context_transform : identity_transform
      |  item_transform : identity_transform
      |  combined_transform : identity_transform
      |}
    """.stripMargin
      .format(lambda, embeddingDim, rankLossType)
  }

  @Test
  def testLowRankLinearRpropUniformRankLoss() = {
    testLowRankLinear(0.1, 32, "uniform", false, 0.8)
  }

  @Test
  def testLowRankLinearRpropNonUniformRankLoss() = {
    testLowRankLinear(0.1, 32, "non_uniform", false, 0.8)
  }

  def testLowRankLinear(lambda : Double,
                        embeddingDim : Int,
                        rankLossType: String,
                        multiLabel : Boolean,
                        expectedCorrect : Double) = {
    val (examples, labels) = TrainingTestHelper.makeSimpleMulticlassClassificationExamples(multiLabel)

    var sc = new SparkContext("local", "LowRankLinearTest")

    try {
      val config = ConfigFactory.parseString(makeConfig(lambda, embeddingDim, rankLossType))

      val input = sc.parallelize(examples)
      val model = LowRankLinearTrainer.train(sc, input, config, "model_config")

      val featureWeightVector = model.getFeatureWeightVector.asScala
      for (wv <- featureWeightVector) {
        log.info(wv.toString())
      }

      val labelWeightVector = model.getLabelWeightVector.asScala
      for (wv <- labelWeightVector) {
        log.info(wv.toString())
      }

      var numCorrect: Int = 0
      for (i <- examples.indices) {
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
