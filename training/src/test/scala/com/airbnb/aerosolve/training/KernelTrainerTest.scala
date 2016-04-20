package com.airbnb.aerosolve.training

import java.io.{StringReader, BufferedWriter, BufferedReader, StringWriter}

import com.airbnb.aerosolve.core.models.ModelFactory
import com.airbnb.aerosolve.core.ModelRecord
import com.typesafe.config.ConfigFactory
import org.apache.spark.SparkContext
import org.junit.Test
import org.slf4j.LoggerFactory
import org.junit.Assert._

import scala.collection.JavaConverters._
import scala.collection.JavaConversions

class KernelTrainerTest {
  val log = LoggerFactory.getLogger("KernelTrainerTest")

  def makeConfig(loss : String, kernel : String) : String = {
    """
      |identity_transform {
      |  transform : list
      |  transforms: []
      |}
      |model_config {
      |  rank_key : "$rank"
      |  loss : "%s"
      |  kernel : "%s"
      |  rank_threshold : 0.0
      |  num_candidates : 1000
      |  max_vectors : 70
      |  min_count : 1
      |  scale : 2.0
      |  learning_rate : 0.1
      |  context_transform : identity_transform
      |  item_transform : identity_transform
      |  combined_transform : identity_transform
      |}
    """.stripMargin
       .format(loss, kernel)
  }

  @Test
  def testRBFHinge() = {
    testKernelClassificationTrainer("hinge", "rbf", 0.7)
  }

  @Test
  def testRBFRegression() = {
    testKernelRegressionTrainer("regression", "rbf", 4.0)
  }

  @Test
   def testAcosHinge() = {
     testKernelClassificationTrainer("hinge", "acos", 0.4)
   }

   @Test
   def testAcosRegression() = {
     testKernelRegressionTrainer("regression", "acos", 8.0)
   }

   @Test
   def testRandomHinge() = {
     testKernelClassificationTrainer("hinge", "random", 0.7)
   }

   @Test
   def testRandomRegression() = {
     testKernelRegressionTrainer("regression", "random", 5.0)
   }

  def testKernelClassificationTrainer(loss : String,
                                      kernel : String,
                                      expectedCorrect : Double) = {
    val (examples, label, numPos) = TrainingTestHelper.makeClassificationExamples

    var sc = new SparkContext("local", "KernelTrainerTest")

    try {
      val config = ConfigFactory.parseString(makeConfig(loss, kernel))

      val input = sc.parallelize(examples)
      val model = KernelTrainer.train(sc, input, config, "model_config")

      var numCorrect : Int = 0
      var i : Int = 0
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
      
      for (sv <- model.getSupportVectors().asScala) {
        log.info(sv.toModelRecord.toString)
      }

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

  def testKernelRegressionTrainer(loss : String,
                                  kernel : String,
                                  maxError : Double) = {
    val (examples, label) = TrainingTestHelper.makeRegressionExamples()

    var sc = new SparkContext("local", "testKernelRegressionTrainerTest")

    try {
      val config = ConfigFactory.parseString(makeConfig(loss, kernel))

      val input = sc.parallelize(examples)
      val model = KernelTrainer.train(sc, input, config, "model_config")

      for (sv <- model.getSupportVectors().asScala) {
        log.info(sv.toModelRecord.toString)
      }

      val labelArr = label.toArray
      var i : Int = 0
      var totalError : Double = 0

      for (ex <- examples) {
        val score = model.scoreItem(ex.example.get(0))
        val exampleLabel = labelArr(i)

        totalError += math.abs(score - exampleLabel)

        i += 1
      }
      log.info("Average absolute error = %f".format(totalError / examples.size.toDouble))
      // Total error not too high
      assertTrue(totalError / examples.size.toDouble < maxError)

    } finally {
      sc.stop
      sc = null

      // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
      System.clearProperty("spark.master.port")
    }
  }

}
