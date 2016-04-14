package com.airbnb.aerosolve.training.pipeline

import com.airbnb.aerosolve.core.features.Family
import com.airbnb.aerosolve.core.{FeatureVector, EvaluationRecord, Example}
import com.airbnb.aerosolve.core.models.AbstractModel
import com.airbnb.aerosolve.core.transforms.Transformer
import com.airbnb.aerosolve.training.LinearRankerUtils
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
      labelFamily: Family,
      useProb: Boolean,
      isMulticlass: Boolean,
      isTraining: Example => Boolean): RDD[EvaluationRecord] = {
    val modelBC = sc.broadcast(modelOpt)
    val transformerBC = sc.broadcast(transformer)
    examples.map(example => exampleToEvaluationRecord(
      example, transformerBC.value,
      modelBC.value, useProb, isMulticlass, labelFamily, isTraining)
    )
  }

  def exampleToEvaluationRecord(
      example: Example,
      transformer: Transformer,
      model: AbstractModel,
      useProb: Boolean,
      isMulticlass: Boolean,
      labelFamily: Family,
      isTraining: Example => Boolean): EvaluationRecord = {
    val result = new EvaluationRecord
    result.setIs_training(isTraining(example))
    example.transform(transformer)

    if (isMulticlass) {
      val score = model.scoreItemMulticlass(example.only).asScala
      val multiclassLabel: FeatureVector = example.only.get(labelFamily)
      val evalScores = new java.util.HashMap[java.lang.String, java.lang.Double]()
      val evalLabels = new java.util.HashMap[java.lang.String, java.lang.Double]()

      result.setScores(evalScores)
      result.setLabels(evalLabels)

      for (s <- score) {
        evalScores.put(s.getLabel, s.getScore)
      }

      for (fv <- multiclassLabel.iterator.asScala) {
        evalLabels.put(fv.feature.name, fv.value)
      }
    } else {
      val score = model.scoreItem(example.only)
      val prob = if (useProb) model.scoreProbability(score) else score
      val rank = LinearRankerUtils.getLabel(example.only, labelFamily)

      result.setScore(prob)
      result.setLabel(rank)
    }

    result
  }

  def scoreExampleForEvaluation(
      sc: SparkContext,
      transformer: Transformer,
      modelOpt: AbstractModel,
      example: Example,
      isTraining: Example => Boolean): EvaluationRecord = {
    val modelBC = sc.broadcast(modelOpt)
    val transformerBC = sc.broadcast(transformer)
    val result = new EvaluationRecord
    result.setIs_training(isTraining(example))

    example.transform(transformerBC.value)
    val score = modelBC.value.scoreItem(example.only)
    val prob = modelBC.value.scoreProbability(score)
    val rank = example.only.get("$rank", "")

    result.setScore(prob)
    result.setLabel(rank)
    result
  }
}