package com.airbnb.aerosolve.training.pipeline

import com.airbnb.aerosolve.training.pipeline.PipelineTestingUtil._
import com.google.common.collect.ImmutableMap
import org.junit.Assert._
import org.junit.Test

class EvalUtilTest {
  @Test
  def testExampleToEvaluationRecordMulticlass() = {
    val evalResult = EvalUtil.exampleToEvaluationRecord(
      multiclassExample1,
      transformer,
      fullRankLinearModel,
      false,
      true,
      "LABEL",
      _ => false
    )

    assertEquals(evalResult.getLabelsSize, 2)
    assertEquals(evalResult.getScoresSize, 2)

    assertEquals(evalResult.getLabels.get("label1"), 10.0, 0.001)
    assertEquals(evalResult.getLabels.get("label2"), 9.0, 0.001)
    assertEquals(evalResult.getScores.get("label1"), 1.2 * 1.2 + 5.6 * 3.4, 0.001)
    assertEquals(evalResult.getScores.get("label2"), 1.2 * 2.1 + 5.6 * -1.2, 0.001)
    assertEquals(evalResult.isIs_training, false)
  }

  @Test
  def testExampleToEvaluationRecordLinear() = {
    val evalResult = EvalUtil.exampleToEvaluationRecord(
      linearExample1,
      transformer,
      linearModel,
      false,
      false,
      "LABEL",
      _ => false
    )

    assertEquals(evalResult.getLabel, 3.5, 0.001)
    assertEquals(evalResult.getScore, 1.4 + 1.3, 0.001)
    assertEquals(evalResult.isIs_training, false)
  }

  @Test
  def testExampleToEvaluationRecordLinearProb() = {
    val evalResult = EvalUtil.exampleToEvaluationRecord(
      linearExample1,
      transformer,
      linearModel,
      true,
      false,
      "LABEL",
      _ => true
    )

    assertEquals(evalResult.getLabel, 3.5, 0.001)
    assertEquals(evalResult.getScore, 1.0 / (1.0 + math.exp(-(1.3 + 1.4))), 0.001)
    assertEquals(evalResult.isIs_training, true)
  }

  @Test
  def testScoreExamplesForEvaluation() = {
    PipelineTestingUtil.withSparkContext(sc => {
      val examples = sc.parallelize(Seq(multiclassExample1, multiclassExample2))

      val results = EvalUtil.scoreExamplesForEvaluation(
        sc, transformer, fullRankLinearModel, examples, "LABEL", false, true, _ => false
      ).collect()

      assertEquals(results.size, 2)
      assertEquals(results(0).getLabels, ImmutableMap.of("label1", 10.0, "label2", 9.0))
      assertEquals(results(1).getLabels, ImmutableMap.of("label1", 8.0, "label2", 4.0))

      // Test one of the results in more detail
      assertEquals(results(0).getScores.get("label1"), 1.2 * 1.2 + 5.6 * 3.4, 0.001)
      assertEquals(results(0).isIs_training, false)
    })
  }
}
