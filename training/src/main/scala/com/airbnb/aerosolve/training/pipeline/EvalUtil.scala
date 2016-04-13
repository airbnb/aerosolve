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

  def scoreExamples(
      sc: SparkContext,
      transformer: Transformer,
      modelOpt: AbstractModel,
      examples: RDD[Example],
      isTraining: Example => Boolean,
      labelKey: String): RDD[(Float, String)] = {
    val modelBC = sc.broadcast(modelOpt)
    val transformerBC = sc.broadcast(transformer)
    val scoreAndLabel = examples
      .map(example => {
        transformerBC.value.combineContextAndItems(example)
        val score = modelBC.value.scoreItem(example.example.get(0))
        val rank = example.example.get(0).floatFeatures.get(labelKey).get("")
        val label = (if (isTraining(example)) "TRAIN_" else "HOLD_") + (if (rank > 0) "P" else "N")
        (score, label)
      })
    scoreAndLabel
  }

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

    transformerBC.value.combineContextAndItems(example)
    val score = modelBC.value.scoreItem(example.example.get(0))
    val prob = modelBC.value.scoreProbability(score)
    val rank = example.example.get(0).floatFeatures.get("$rank").get("")

    result.setScore(prob)
    result.setLabel(rank)
    result
  }

  def getClassificationAUC(records : Seq[EvaluationRecord]) : Double = {
    // Find minimal and maximal scores
    var minScore = records.head.score
    var maxScore = minScore

    records.foreach(record => {
      val score = record.score

      minScore = Math.min(minScore, score)
      maxScore = Math.max(maxScore, score)
    })

    if (minScore >= maxScore) {
      log.warn("max score smaller than or equal to min score (%f, %f).".format(minScore, maxScore))
      maxScore = minScore + 1.0
    }

    // For AUC evaluation
    val buckets = records
      .map(x => evaluateRecordForAUC(x, minScore, maxScore))
      .groupBy(_._1)
      .map(x => x._2.reduce((a, b) => (a._1, (a._2._1 + b._2._1, a._2._2 + b._2._2))))
      .toArray
      .sortBy(x => x._1)

    getAUC(buckets.map(x => (x._2._1, x._2._2)))
  }

  private def evaluateRecordForAUC(
      record : EvaluationRecord,
      minScore : Double,
      maxScore : Double) : (Long, (Long, Long)) = {
    var offset = 0

    if (record.label <= 0) {
      offset += 1
    }

    val score : Long = ((record.score - minScore) / (maxScore - minScore) * 100).toLong

    offset match {
      case 0 => (score, (1, 0))
      case 1 => (score, (0, 1))
    }
  }

  private def getAUC(buckets: Array[(Long, Long)]): Double = {
    val tot = buckets.reduce((x, y) => (x._1 + y._1, x._2 + y._2))
    var auc = 0.0
    var cs=(0L, 0L)
    for (x <- buckets) {
      // area for the current slice: (TP0+TP1)/2*(TN1-TN0)
      auc += ((tot._1 - cs._1) + (tot._1 - cs._1 - x._1)) / 2.0 * x._2
      // (TP0, TN0) -> (TP1, TN1)
      cs = (cs._1 + x._1, cs._2 + x._2)
    }
    auc / tot._1 / tot._2
  }
}