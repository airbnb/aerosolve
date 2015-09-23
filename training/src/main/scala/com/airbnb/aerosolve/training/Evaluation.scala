package com.airbnb.aerosolve.training

import org.slf4j.{Logger, LoggerFactory}
import org.apache.spark.rdd.RDD
import com.airbnb.aerosolve.core.EvaluationRecord
import org.apache.spark.SparkContext._
import scala.collection.mutable.{ArrayBuffer, Buffer}

/*
* Given an RDD of EvaluationRecord return standard evaluation metrics
*/

object Evaluation {
  private final val log: Logger = LoggerFactory.getLogger("Evaluation")

  def evaluateBinaryClassification(records : RDD[EvaluationRecord],
                                   buckets : Int,
                                   evalMetric : String) : Array[(String, Double)] = {
    var metrics  = Buffer[(String, Double)]()
    var bestF1 = -1.0
    val thresholds = records.map(x => x.score).histogram(buckets)._1
    // Search all thresholds for the best F1
    // At the same time collect the precision and recall.
    val trainPR = new ArrayBuffer[(Double, Double)]()
    val holdPR = new ArrayBuffer[(Double, Double)]()
    trainPR.append((1.0, 0.0))
    holdPR.append((1.0, 0.0))
    for (i <- 0 until thresholds.size - 1) {
      val threshold = thresholds(i)
      val tmp = evaluateBinaryClassificationAtThreshold(records, threshold)
      val tmpMap = tmp.toMap
      val f1 = tmpMap.getOrElse(evalMetric, 0.0)
      if (f1 > bestF1) {
        bestF1 = f1
        metrics = tmp
      }
      trainPR.append((tmpMap.getOrElse("!TRAIN_PRECISION", 0.0), tmpMap.getOrElse("!TRAIN_RECALL", 0.0)))
      holdPR.append((tmpMap.getOrElse("!HOLD_PRECISION", 0.0), tmpMap.getOrElse("!HOLD_RECALL", 0.0)))
    }

    metrics.append(("!TRAIN_PR_AUC", getPRAUC(trainPR)))
    metrics.append(("!HOLD_PR_AUC", getPRAUC(holdPR)))
    for (tpr <- trainPR) {
      metrics.append(("!TRAIN_PREC@RECALL=%f".format(tpr._2), tpr._1))
    }
    for (hpr <- holdPR) {
      metrics.append(("!HOLD_PREC@RECALL=%f".format(hpr._2), hpr._1))
    }

    evaluateBinaryClassificationAUC(records, metrics)

    metrics
      .sortWith((a, b) => a._1 < b._1)
      .toArray
  }

  private def evaluateBinaryClassificationAtThreshold(records : RDD[EvaluationRecord],
                                                      threshold : Double) : Buffer[(String, Double)] = {
    val metricsMap = records
      .flatMap(x => evaluateRecordBinaryClassification(x, threshold))
      .reduceByKey(_ + _)
      .collectAsMap

    val metrics = metricsMap.toBuffer

    val ttp = metricsMap.getOrElse("TRAIN_TP", 0.0)
    val ttn = metricsMap.getOrElse("TRAIN_TN", 0.0)
    val tfp = metricsMap.getOrElse("TRAIN_FP", 0.0)
    val tfn = metricsMap.getOrElse("TRAIN_FN", 0.0)

    val te = metricsMap.getOrElse("TRAIN_SQERR", 0.0)
    val tc = metricsMap.getOrElse("TRAIN_COUNT", 0.0)

    metrics.append(("!BEST_THRESHOLD", threshold))
    metrics.append(("!TRAIN_ACC", (ttp + ttn) / tc))
    metrics.append(("!TRAIN_RMSE", Math.sqrt(te / tc)))
    metrics.append(("!TRAIN_FPR", tfp / (tfp + ttn)))
    metrics.append(("!TRAIN_RECALL", if (ttp > 0.0) ttp / (ttp + tfn) else 0.0))
    metrics.append(("!TRAIN_PRECISION", if (ttp > 0.0) ttp / (ttp + tfp) else 0.0))
    metrics.append(("!TRAIN_F1", 2 * ttp / (2 * ttp + tfn + tfp)))

    val htp = metricsMap.getOrElse("HOLD_TP", 0.0)
    val htn = metricsMap.getOrElse("HOLD_TN", 0.0)
    val hfp = metricsMap.getOrElse("HOLD_FP", 0.0)
    val hfn = metricsMap.getOrElse("HOLD_FN", 0.0)

    val he = metricsMap.getOrElse("HOLD_SQERR", 0.0)
    val hc = metricsMap.getOrElse("HOLD_COUNT", 0.0)

    metrics.append(("!HOLD_ACC", (htp + htn) / hc))
    metrics.append(("!HOLD_RMSE", Math.sqrt(he / hc)))
    metrics.append(("!HOLD_FPR", hfp / (hfp + htn)))
    metrics.append(("!HOLD_RECALL", if (htp > 0.0) htp / (htp + hfn) else 0.0))
    metrics.append(("!HOLD_PRECISION", if (htp > 0.0) htp / (htp + hfp) else 0.0))
    metrics.append(("!HOLD_F1", 2 * htp / (2 * htp + hfn + hfp)))
    metrics
  }

  def evaluateRegression(records : RDD[EvaluationRecord]) : Array[(String, Double)] = {
    val metricsMap = records
      .flatMap(x => evaluateRecordRegression(x))
      .reduceByKey(_ + _)
      .collectAsMap

    val metrics = metricsMap.toBuffer
    val te = metricsMap.getOrElse("TRAIN_SQERR", 0.0)
    val tc = metricsMap.getOrElse("TRAIN_COUNT", 0.0)
    metrics.append(("!TRAIN_RMSE", Math.sqrt(te / tc)))
    val he = metricsMap.getOrElse("HOLD_SQERR", 0.0)
    val hc = metricsMap.getOrElse("HOLD_COUNT", 0.0)
    metrics.append(("!HOLD_RMSE", Math.sqrt(he / hc)))
    metrics
      .sortWith((a, b) => a._1 < b._1)
      .toArray
  }

  private def evaluateBinaryClassificationAUC(records : RDD[EvaluationRecord],
                                              metrics : Buffer[(String, Double)]) = {
    val COUNT = 5

    // Partition the records into COUNT independent populations
    val recs = records.map(x => (scala.util.Random.nextInt(COUNT),x))

    // Sum and sum squared AUC, min, max
    var train = (0.0, 0.0, 1e10, -1e10)
    var hold = (0.0, 0.0, 1e10, -1e10)
    var count : Double = 0.0
    for (i <- 0 until COUNT) {
      // Result contains (Train AUC, Hold AUC)
      val myrecs = recs.filter(x => x._1 == i).map(x => x._2)
      val result = getClassificationAUCTrainHold(myrecs)
      train = (train._1 + result._1, train._2 + result._1 * result._1,
        Math.min(train._3, result._1), Math.max(train._4, result._1))
      hold =  (hold._1 + result._2, hold._2 + result._2 * result._2,
        Math.min(hold._3, result._2), Math.max(hold._4, result._2))
    }

    val trainMean = train._1 / COUNT.toDouble
    val trainVar = (train._2) / COUNT.toDouble - trainMean * trainMean
    metrics.append(("!TRAIN_AUC", trainMean))
    metrics.append(("!TRAIN_AUC_STDDEV", Math.sqrt(trainVar)))
    metrics.append(("!TRAIN_AUC_MIN", train._3))
    metrics.append(("!TRAIN_AUC_MAX", train._4))
    val holdMean = hold._1 / COUNT.toDouble
    val holdVar = (hold._2) / COUNT.toDouble - holdMean * holdMean
    metrics.append(("!HOLD_AUC", holdMean))
    metrics.append(("!HOLD_AUC_STDDEV", Math.sqrt(holdVar)))
    metrics.append(("!HOLD_AUC_MIN", hold._3))
    metrics.append(("!HOLD_AUC_MAX", hold._4))
  }

  private def getClassificationAUCTrainHold(records : RDD[EvaluationRecord]) : (Double, Double) = {
    // find minimal and maximal scores
    var minScore = records.take(1).apply(0).score
    var maxScore = minScore
    records.foreach(record => {
      val score = record.score
      minScore = Math.min(minScore, score)
      maxScore = Math.max(maxScore, score)
    })

    if(minScore >= maxScore) {
      log.warn("max score smaller than or equal to min score (%f, %f).".format(minScore, maxScore))
      maxScore = minScore + 1.0
    }

    // for AUC evaluation
    val buckets = records
      .map(x => evaluateRecordForAUC(x, minScore, maxScore))
      .reduceByKey((a,b) => (a + b))
      .sortBy(x => x._1)
      .collect

    (getAUC(buckets.map(x=>(x._2._1, x._2._2))), getAUC(buckets.map(x=>(x._2._3, x._2._4))))
  }

  implicit class Tupple4Add(t: (Long, Long, Long, Long)) {
    def +(p: (Long, Long, Long, Long)) = (p._1 + t._1, p._2 + t._2, p._3 + t._3, p._4 + t._4)
  }

  implicit class Tupple2Add(t: (Long, Long)) {
    def +(p: (Long, Long)) = (p._1 + t._1, p._2 + t._2)
  }

  // input is a list of (true positive, true negative) bucketized
  // by ranker output scores in ascending order
  private def getAUC(buckets: Array[(Long, Long)]): Double = {
    val tot = buckets.reduce((x,y)=>(x+y))
    var auc = 0.0
    var cs=(0L, 0L)
    for (x <- buckets) {
      // area for the current slice: (TP0+TP1)/2*(TN1-TN0)
      auc += ((tot._1 - cs._1) + (tot._1 - cs._1 - x._1)) / 2.0 * x._2
      // (TP0, TN0) -> (TP1, TN1)
      cs += x
    }
    auc / tot._1 / tot._2
  }

  // Uses trapezium rule to compute the precision recall AUC.
  private def getPRAUC(pr : Buffer[(Double, Double)]) : Double = {
    val sorted = pr.sortWith((a, b) => a._2 < b._2)
    val count = sorted.size
    var sum = 0.0
    for (i <- 1 until count) {
      val dR = sorted(i)._2 - sorted(i - 1)._2
      val dP = 0.5 * (sorted(i)._1 + sorted(i - 1)._1)
      sum += dR * dP
    }
    sum
  }

  private def evaluateRecordForAUC(record : EvaluationRecord,
                                   minScore : Double,
                                   maxScore : Double) : (Long, (Long, Long, Long, Long)) = {
    var offset = if (record.is_training) 0 else 2
    if (record.label <= 0) {
      offset += 1
    }

    val score : Long = ((record.score - minScore) / (maxScore - minScore) * 100).toLong

    offset match {
      case 0 => (score, (1, 0, 0, 0))
      case 1 => (score, (0, 1, 0, 0))
      case 2 => (score, (0, 0, 1, 0))
      case 3 => (score, (0, 0, 0, 1))
    }
  }

  private def evaluateRecordRegression(record : EvaluationRecord) : Iterator[(String, Double)] = {
    val out = collection.mutable.ArrayBuffer[(String, Double)]()

    val prefix = if (record.is_training) "TRAIN_" else "HOLD_"
    out.append((prefix + "SQERR", (record.label - record.score) * (record.label - record.score)))
    out.append((prefix + "COUNT", 1.0))
    out.iterator
  }

  // Given a record and a threshold compute if it is a true/false positive/negative
  // and the squared error assuming it is a probability.
  private def evaluateRecordBinaryClassification(record : EvaluationRecord,
                                                 threshold : Double) : Iterator[(String, Double)] = {
    val out = collection.mutable.ArrayBuffer[(String, Double)]()

    val prefix = if (record.is_training) "TRAIN_" else "HOLD_"
    if (record.score > threshold) {
      if (record.label > 0) {
        out.append((prefix + "TP", 1.0))
      } else {
        out.append((prefix + "FP", 1.0))
      }
    } else {
      if (record.label <= 0) {
        out.append((prefix + "TN", 1.0))
      } else {
        out.append((prefix + "FN", 1.0))
      }
    }

    val error = if (record.label > 0) {
      (1.0 - record.score) * (1.0 - record.score)
    } else {
      record.score * record.score
    }
    out.append((prefix + "SQERR", error))
    out.append((prefix + "COUNT", 1.0))
    out.iterator
  }
}