package com.airbnb.aerosolve.training.pipeline

import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.training.pipeline.PipelineTestingUtil._
import org.junit.Assert._
import org.junit.Test

class PipelineUtilTest {
  @Test
  def testScoreExamples() = {
    PipelineTestingUtil.withSparkContext(sc => {
      val examples = sc.parallelize(Seq(linearExample1, linearExample2))

      val trainingPredicate = (example: Example) => {
        example.getExample.get(0).getStringFeatures.get("s").size() == 1
      }

      val results = PipelineUtil.scoreExamples(
        sc, transformer, linearModel, examples, trainingPredicate, "LABEL"
      ).collect()

      assertEquals(results.size, 2)
      assertEquals(results(0)._1, 1.4 + 1.3, 0.001)
      assertEquals(results(0)._2, "HOLD_P")

      assertEquals(results(1)._1, 1.4, 0.001)
      assertEquals(results(1)._2, "TRAIN_N")
    })
  }
}
