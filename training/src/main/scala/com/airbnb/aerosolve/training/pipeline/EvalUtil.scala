package com.airbnb.aerosolve.training.pipeline

import com.airbnb.aerosolve.core.{EvaluationRecord, Example}
import com.airbnb.aerosolve.core.models.AbstractModel
import com.airbnb.aerosolve.core.transforms.Transformer
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._

/*
 * Various utilities used in the GenericPipeline for model evaluation.
 */
object EvalUtil {
  val log: Logger = LoggerFactory.getLogger(this.getClass)

  def scoreExamplesForEvaluation(
      sc: SparkContext,
      transformer: Transformer,
      modelOpt: AbstractModel,
      examples: RDD[Example],
      label: String,
      useProb: Boolean,
      isMulticlass: Boolean,
      isTraining: Example => Boolean): RDD[EvaluationRecord] = {
    val modelBC = sc.broadcast(modelOpt)
    val transformerBC = sc.broadcast(transformer)
    examples.map(example => exampleToEvaluationRecord(
      example, transformerBC.value,
      modelBC.value, useProb, isMulticlass, label, isTraining)
    )
  }

  def exampleToEvaluationRecord(
      example: Example,
      transformer: Transformer,
      model: AbstractModel,
      useProb: Boolean,
      isMulticlass: Boolean,
      label: String,
      isTraining: Example => Boolean): EvaluationRecord = {
    val result = new EvaluationRecord
    result.setIs_training(isTraining(example))
    transformer.combineContextAndItems(example)

    if (isMulticlass) {
      val score = model.scoreItemMulticlass(example.example.get(0)).asScala
      val multiclassLabel = example.example.get(0).floatFeatures.get(label).asScala
      val evalScores = new java.util.HashMap[java.lang.String, java.lang.Double]()
      val evalLabels = new java.util.HashMap[java.lang.String, java.lang.Double]()

      result.setScores(evalScores)
      result.setLabels(evalLabels)

      for (s <- score) {
        evalScores.put(s.label, s.score)
      }

      for (l <- multiclassLabel) {
        evalLabels.put(l._1, l._2)
      }
    } else {
      val score = model.scoreItem(example.example.get(0))
      val prob = if (useProb) model.scoreProbability(score) else score
      val rank = example.example.get(0).floatFeatures.get(label).values().iterator().next()

      result.setScore(prob)
      result.setLabel(rank)
    }

    result
  }
}