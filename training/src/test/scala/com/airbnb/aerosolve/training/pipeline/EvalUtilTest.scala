package com.airbnb.aerosolve.training.pipeline

import com.airbnb.aerosolve.core.{FeatureVector, Example, LabelDictionaryEntry}
import com.airbnb.aerosolve.core.models.{LinearModel, FullRankLinearModel}
import com.airbnb.aerosolve.core.transforms.Transformer
import com.airbnb.aerosolve.core.util.FloatVector
import com.google.common.collect.{ImmutableMap, ImmutableSet}
import com.typesafe.config.ConfigFactory
import org.junit.Assert._
import org.junit.Test

class EvalUtilTest {
  val transformer = {
    val config = """
       |identity_transform {
       |  transform : list
       |  transforms : [ ]
       |}
       |
       |model_transforms {
       |  context_transform : identity_transform
       |  item_transform : identity_transform
       |  combined_transform : identity_transform
       |}
     """.stripMargin

    new Transformer(ConfigFactory.parseString(config), "model_transforms")
  }

  // Simple full rank linear model with 2 label classes and 2 features
  val fullRankLinearModel = {
    val model = new FullRankLinearModel()

    model.setLabelToIndex(ImmutableMap.of("label1", 0, "label2", 1))

    val labelDictEntry1 = new LabelDictionaryEntry()
    labelDictEntry1.setLabel("label1")
    labelDictEntry1.setCount(50)

    val labelDictEntry2 = new LabelDictionaryEntry()
    labelDictEntry2.setLabel("label2")
    labelDictEntry2.setCount(100)

    val labelDictionary = new java.util.ArrayList[LabelDictionaryEntry]()

    labelDictionary.add(labelDictEntry1)
    labelDictionary.add(labelDictEntry2)

    model.setLabelDictionary(labelDictionary)

    val floatVector1 = new FloatVector(Array(1.2f, 2.1f))
    val floatVector2 = new FloatVector(Array(3.4f, -1.2f))

    model.setWeightVector(
      ImmutableMap.of(
        "f", ImmutableMap.of("feature1", floatVector1, "feature2", floatVector2)
      )
    )

    model
  }

  // Simple linear model with 2 features
  val linearModel = {
    val model = new LinearModel()

    model.setWeights(ImmutableMap.of("s", ImmutableMap.of("feature1", 1.4f, "feature2", 1.3f)))

    model
  }

  val multiclassExample1 = {
    val example = new Example()
    val fv = new FeatureVector()

    fv.setFloatFeatures(ImmutableMap.of(
      "f", ImmutableMap.of("feature1", 1.2, "feature2", 5.6),
      "LABEL", ImmutableMap.of("label1", 10.0, "label2", 9.0)
    ))

    example.addToExample(fv)

    example
  }

  val multiclassExample2 = {
    val example = new Example()
    val fv = new FeatureVector()

    fv.setFloatFeatures(ImmutableMap.of(
      "f", ImmutableMap.of("feature1", 1.8, "feature2", -1.6),
      "LABEL", ImmutableMap.of("label1", 8.0, "label2", 4.0)
    ))

    example.addToExample(fv)

    example
  }

  val linearExample1 = {
    val example = new Example()
    val fv = new FeatureVector()

    fv.setFloatFeatures(ImmutableMap.of(
      "LABEL", ImmutableMap.of("", 3.5)
    ))

    fv.setStringFeatures(ImmutableMap.of(
      "s", ImmutableSet.of("feature1", "feature2")
    ))

    example.addToExample(fv)

    example
  }

  val linearExample2 = {
    val example = new Example()
    val fv = new FeatureVector()

    fv.setFloatFeatures(ImmutableMap.of(
      "LABEL", ImmutableMap.of("", -2.0)
    ))

    fv.setStringFeatures(ImmutableMap.of(
      "s", ImmutableSet.of("feature1")
    ))

    example.addToExample(fv)

    example
  }

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
  def testScoreExamples() = {
    PipelineTestingUtil.withSparkContext(sc => {
      val examples = sc.parallelize(Seq(linearExample1, linearExample2))

      val trainingPredicate = (example: Example) => {
        example.getExample.get(0).getStringFeatures.get("s").size() == 1
      }

      val results = EvalUtil.scoreExamples(
        sc, transformer, linearModel, examples, trainingPredicate, "LABEL"
      ).collect()

      assertEquals(results.size, 2)
      assertEquals(results(0)._1, 1.4 + 1.3, 0.001)
      assertEquals(results(0)._2, "HOLD_P")

      assertEquals(results(1)._1, 1.4, 0.001)
      assertEquals(results(1)._2, "TRAIN_N")
    })
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
