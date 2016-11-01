package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core.EvaluationRecord
import com.airbnb.aerosolve.training.pipeline.ResultUtil
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.collection.{Map, mutable}

/*
* Given an RDD of EvaluationRecord return standard evaluation metrics
*/

object Evaluation {
  private final val log: Logger = LoggerFactory.getLogger("Evaluation")

  def evaluateBinaryClassification(records: List[EvaluationRecord],
                                   buckets: Int,
                                   evalMetric: String): Array[(String, Double)] = {
    evaluateBinaryClassificationWithResults(records, buckets, evalMetric, null)
  }

  def evaluateBinaryClassification(records: RDD[EvaluationRecord],
                                   buckets: Int,
                                   evalMetric: String): Array[(String, Double)] = {
    evaluateBinaryClassificationWithResults(records, buckets, evalMetric, null)
  }

  def evaluateBinaryClassificationWithResults(records: List[EvaluationRecord],
                                              buckets: Int,
                                              evalMetric: String,
                                              resultsOutputPath: String): Array[(String, Double)] = {
    val scores = records.map(x => x.score)
    val thresholds = getThresholds(scores.min, scores.max, buckets)

    evaluateBinaryClassificationWithResults(
      thresholds,
      evaluateBinaryClassificationAtThreshold(records, _),
      evaluateBinaryClassificationAUC(records, _),
      buckets,
      evalMetric,
      resultsOutputPath
    )
  }

  def evaluateBinaryClassificationWithResults(records: RDD[EvaluationRecord],
                                              buckets: Int,
                                              evalMetric: String,
                                              resultsOutputPath: String): Array[(String, Double)] = {
    val scores: RDD[Double] = records.map(x => x.score)
    val thresholds = scores.histogram(buckets)._1

    evaluateBinaryClassificationWithResults(
      thresholds,
      evaluateBinaryClassificationAtThreshold(records, _),
      evaluateBinaryClassificationAUC(records, _),
      buckets,
      evalMetric,
      resultsOutputPath
    )
  }

  def evaluateBinaryClassificationWithResults(thresholds: Array[Double],
                                              evaluateBinaryClassificationAtThreshold: Double => mutable.Buffer[(String, Double)],
                                              evaluateBinaryClassificationAUC: mutable.Buffer[(String, Double)] => Unit,
                                              buckets: Int,
                                              evalMetric: String,
                                              resultsOutputPath: String): Array[(String, Double)] = {

    var metrics = mutable.Buffer[(String, Double)]()
    var bestF1 = -1.0
    val holdThresholdPrecisionRecall = mutable.Buffer[(Double, Double, Double)]()
    val trainThresholdPrecisionRecall = mutable.Buffer[(Double, Double, Double)]()

    // Search all thresholds for the best F1
    // At the same time collect the precision and recall.
    val trainPR = new ArrayBuffer[(Double, Double, Double)]()
    val holdPR = new ArrayBuffer[(Double, Double, Double)]()
    trainPR.append((1.0, 0.0, thresholds.max))
    holdPR.append((1.0, 0.0, thresholds.max))

    for (i <- 0 until thresholds.length - 1) {
      val threshold = thresholds(i)
      val tmp = evaluateBinaryClassificationAtThreshold(threshold)
      val tmpMap = tmp.toMap
      val f1 = tmpMap.getOrElse(evalMetric, 0.0)
      if (f1 > bestF1) {
        bestF1 = f1
        metrics = tmp
      }
      trainPR.append((tmpMap.getOrElse("!TRAIN_PRECISION", 0.0), tmpMap.getOrElse("!TRAIN_RECALL", 0.0), threshold))
      holdPR.append((tmpMap.getOrElse("!HOLD_PRECISION", 0.0), tmpMap.getOrElse("!HOLD_RECALL", 0.0), threshold))
    }

    metricsAppend(metrics, trainPR, holdPR, holdThresholdPrecisionRecall, trainThresholdPrecisionRecall)
    evaluateBinaryClassificationAUC(metrics)

    metrics.sortWith((a, b) => a._1 < b._1)

    // Write results
    if (resultsOutputPath != null) {
      ResultUtil.writeResults(resultsOutputPath, metrics.toArray, holdThresholdPrecisionRecall.toArray, trainThresholdPrecisionRecall.toArray)
    }

    // Format threshold metrics into Strings
    for (tpr <- trainThresholdPrecisionRecall) {
      metrics.append(("!TRAIN_THRESHOLD=%f PRECISION=%f RECALL=%f".format(tpr._1, tpr._2, tpr._3), 0))
    }
    for (hpr <- holdThresholdPrecisionRecall) {
      metrics.append(("!HOLD_THRESHOLD=%f PRECISION=%f RECALL=%f".format(hpr._1, hpr._2, hpr._3), 0))
    }

    metrics.sortWith((a, b) => a._1 < b._1).toArray
  }

  def evaluateMulticlassClassification(records: RDD[EvaluationRecord]): Array[(String, Double)] = {
    records.flatMap(rec => {
      // Metric, value, count
      val metrics = scala.collection.mutable.ArrayBuffer[(String, (Double, Double))]()
      if (rec.scores != null && rec.labels != null) {
        val prefix = if (rec.is_training) "TRAIN_" else "HOLD_"
        // Order by top scores.
        val sorted = rec.scores.asScala.toBuffer.sortWith((a, b) => a._2 > b._2)
        // All pairs hinge loss
        val count = sorted.size
        var hingeLoss = 0.0
        for (label <- rec.labels.asScala) {
          for (j <- 0 until count) {
            if (label._1 != sorted(j)._1) {
              val scorei = rec.scores.get(label._1)
              val scorej = sorted(j)._2
              val truei = label._2
              var truej = rec.labels.get(sorted(j)._1)
              if (truej == null) truej = 0.0
              if (truei > truej) {
                val margin = truei - truej
                hingeLoss = hingeLoss + math.max(0.0, margin - scorei + scorej)
              } else if (truei < truej) {
                val margin = truej - truei
                hingeLoss = hingeLoss + math.max(0.0, margin - scorej + scorei)
              }
            }
          }
        }
        metrics.append((prefix + "ALL_PAIRS_HINGE_LOSS", (hingeLoss, 1.0)))
        var inTopK = false
        var nAccurateLabel = 0.0
        var sumPrecision = 0.0
        for (i <- sorted.indices) {
          if (rec.labels.containsKey(sorted(i)._1)) {
            inTopK = true
            nAccurateLabel += 1
            sumPrecision += nAccurateLabel / (i + 1)
            metrics.append((prefix + "MEAN_RECIPROCAL_RANK", (1.0 / (i + 1), 1.0)))
          }
          metrics.append((prefix + "PRECISION@" + (i + 1), (nAccurateLabel / (i + 1), 1.0)))
          metrics.append((prefix + "ACCURACY@" + (i + 1), (if (inTopK) 1.0 else 0.0, 1.0)))
          val denoMAP = math.min(rec.labels.size, i + 1)
          metrics.append((prefix + "MEAN_AVERAGE_PRECISION@" + (i + 1), (sumPrecision / denoMAP, 1.0)))
        }
      }
      metrics
    })
      // Sum all the doubles
      .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
      // Average the values
      .map(x => (x._1, x._2._1 / x._2._2))
      .collect
      .sortWith((a, b) => a._1 < b._1)
  }

  private def metricsAppend(metrics: mutable.Buffer[(String, Double)],
                            trainPR: ArrayBuffer[(Double, Double, Double)],
                            holdPR: ArrayBuffer[(Double, Double, Double)],
                            trainThresholdPrecisionRecall: mutable.Buffer[(Double, Double, Double)],
                            holdThresholdPrecisionRecall: mutable.Buffer[(Double, Double, Double)]): Unit = {
    metrics.append(("!TRAIN_PRECISION_RECALL_AUC", getPRAUC(trainPR)))
    metrics.append(("!HOLD_PRECISION_RECALL_AUC", getPRAUC(holdPR)))

    for (tpr <- trainPR) {
      trainThresholdPrecisionRecall.append((tpr._3, tpr._1, tpr._2))
    }
    for (hpr <- holdPR) {
      holdThresholdPrecisionRecall.append((hpr._3, hpr._1, hpr._2))
    }
  }

  private def evaluateBinaryClassificationAtThreshold(records: List[EvaluationRecord],
                                                      threshold: Double): mutable.Buffer[(String, Double)] = {
    val metricsMap = records
      .flatMap(evaluateRecordBinaryClassification(threshold))
      .groupBy(_._1)
      .map {
        case (_, recs) => recs.reduce((a, b) => (a._1, a._2 + b._2))
      }
    appendMetrics(metricsMap, threshold)
  }

  private def evaluateBinaryClassificationAtThreshold(records: RDD[EvaluationRecord],
                                                      threshold: Double): mutable.Buffer[(String, Double)] = {
    val metricsMap = records
      .flatMap(evaluateRecordBinaryClassification(threshold))
      .reduceByKey(_ + _)
      .collectAsMap()
    appendMetrics(metricsMap, threshold)
  }

  private def appendMetrics(metricsMap: Map[String, Double], threshold: Double): mutable.Buffer[(String, Double)] = {
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

  private def regressionMetrics(map: Map[String, Double], prefix: String,
                                metrics: mutable.Buffer[(String, Double)]): Unit = {
    val te = map.getOrElse(prefix + "SQERR", 0.0)
    val tc = map.getOrElse(prefix + "COUNT", 0.0)
    // to compute SMAPE third version https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    val absLabelMinusScore = map.getOrElse(prefix + "ABS_LABEL_MINUS_SCORE", 0.0)
    val labelPlusScore = map.getOrElse(prefix + "LABEL_PLUS_SCORE", 0.0)
    metrics.append(("!" + prefix + "RMSE", Math.sqrt(te / tc)))
    metrics.append(("!" + prefix + "SMAPE", absLabelMinusScore / labelPlusScore))
  }

  def evaluateRegression(records: RDD[EvaluationRecord]): Array[(String, Double)] = {
    val metricsMap = records
      .flatMap(evaluateRecordRegression)
      .reduceByKey(_ + _)
      .collectAsMap

    val metrics = metricsMap.toBuffer
    regressionMetrics(metricsMap, "TRAIN_", metrics)
    regressionMetrics(metricsMap, "HOLD_", metrics)
    metrics
      .sortWith((a, b) => a._1 < b._1)
      .toArray
  }

  private def evaluateBinaryClassificationAUC(records: List[EvaluationRecord],
                                              metrics: mutable.Buffer[(String, Double)]): Unit = {
    val COUNT = 5
    // Partition the records into COUNT independent populations
    val recs = records.map(x => (scala.util.Random.nextInt(COUNT), x))
    evaluateBinaryClassificationAUC(
      i => {
        val myRecs = recs.filter(x => x._1 == i).map(x => x._2)
        getClassificationAUCTrainHold(myRecs)
      },
      metrics
    )
  }

  private def evaluateBinaryClassificationAUC(records: RDD[EvaluationRecord],
                                              metrics: mutable.Buffer[(String, Double)]): Unit = {
    val COUNT = 5
    // Partition the records into COUNT independent populations
    val recs = records.map(x => (scala.util.Random.nextInt(COUNT), x)).cache()
    evaluateBinaryClassificationAUC(
      i => {
        val myRecs = recs.filter(x => x._1 == i).map(x => x._2)
        getClassificationAUCTrainHold(myRecs)
      },
      metrics
    )
    recs.unpersist()
  }

  private def evaluateBinaryClassificationAUC(getClassificationAUCTrainHold: Int => (Double, Double),
                                              metrics: mutable.Buffer[(String, Double)]): Unit = {
    val COUNT = 5

    // Sum and sum squared AUC, min, max
    var train = (0.0, 0.0, 1e10, -1e10)
    var hold = (0.0, 0.0, 1e10, -1e10)
    for (i <- 0 until COUNT) {
      // Result contains (Train AUC, Hold AUC)
      val result = getClassificationAUCTrainHold(i)
      train = (train._1 + result._1, train._2 + result._1 * result._1,
        Math.min(train._3, result._1), Math.max(train._4, result._1))
      hold = (hold._1 + result._2, hold._2 + result._2 * result._2,
        Math.min(hold._3, result._2), Math.max(hold._4, result._2))
    }

    updateMetricsForAUC(metrics, train, hold, COUNT.toDouble)
  }

  private def updateMetricsForAUC(metrics: mutable.Buffer[(String, Double)],
                                  train: (Double, Double, Double, Double),
                                  hold: (Double, Double, Double, Double),
                                  count: Double,
                                  results: mutable.Buffer[(String, Double)] = null) = {
    val trainMean = train._1 / count
    val trainVar = train._2 / count - trainMean * trainMean
    metrics.append(("!TRAIN_AUC", trainMean))
    metrics.append(("!TRAIN_AUC_STDDEV", Math.sqrt(trainVar)))
    metrics.append(("!TRAIN_AUC_MIN", train._3))
    metrics.append(("!TRAIN_AUC_MAX", train._4))

    val holdMean = hold._1 / count
    val holdVar = hold._2 / count - holdMean * holdMean
    metrics.append(("!HOLD_AUC", holdMean))
    metrics.append(("!HOLD_AUC_STDDEV", Math.sqrt(holdVar)))
    metrics.append(("!HOLD_AUC_MIN", hold._3))
    metrics.append(("!HOLD_AUC_MAX", hold._4))
  }

  private def getClassificationAUCTrainHold(records: RDD[EvaluationRecord]): (Double, Double) = {
    // find minimal and maximal scores
    val scores = records.map(r => (r.score, r.score))
    val (maxScore, minScore) = scores.reduce( {case (u, v) => (u._1 max v._1, u._2 min v._2) })
    val count = records.count()
    val bucketSize = Math.max(1, count / 10000) // Divide the area under ROC curve by 10000 vertical strips
    log.info("%d eval record, bucket size = %d".format(count, bucketSize))
    if (minScore >= maxScore) {
      log.error("max score smaller than or equal to min score (%f, %f), total: %d".
        format(minScore, maxScore, count))
      throw new Exception("%d evaluation records all have same score, something must be wrong".
        format(count))
    }

    // for AUC evaluation
    val buckets = records
      .map(x => evaluateRecordForAUC(x, minScore, maxScore))
      .sortByKey()
      .zipWithIndex()
      .map({case ((score, tuple), idx) => (idx / bucketSize, tuple)})
      .reduceByKey((a, b) => a + b)
      .sortBy(x => x._1)
      .collect

    (getAUC(buckets.map(x => (x._2._1, x._2._2))), getAUC(buckets.map(x => (x._2._3, x._2._4))))
  }

  def getClassificationAUC(records: List[EvaluationRecord]): Double = {
    // compute just one AUC for all records without discriminating training and hold out
    val evalRecords = records.map(record => {
      // assuming all in holdout group
      record.setIs_training(false)
      record
    })
    val aucs = getClassificationAUCTrainHold(evalRecords)
    aucs._2
  }

  private def getClassificationAUCTrainHold(records: List[EvaluationRecord]): (Double, Double) = {
    // find minimal and maximal scores
    val scores = records.map(rec => rec.score)
    val minScore = scores.min
    var maxScore = scores.max
    if (minScore >= maxScore) {
      log.warn("max score smaller than or equal to min score (%f, %f). Number of records: %d".
        format(minScore, maxScore, scores.size))
      maxScore = minScore + 1.0
    }

    // for AUC evaluation
    val buckets = records
      .map(x => evaluateRecordForAUC(x, minScore, maxScore))
      .groupBy(_._1)
      .map { case (key, value) => (key, value.map(x => x._2).reduce { (a, b) => a + b }) }
      .toArray
      .sortBy(x => x._1)
    (getAUC(buckets.map(x => (x._2._1, x._2._2))), getAUC(buckets.map(x => (x._2._3, x._2._4))))
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
    val tot = buckets.reduce((x, y) => x + y)
    var auc = 0.0
    var cs = (0L, 0L)
    for (x <- buckets) {
      // area for the current slice: (TP0+TP1)/2*(TN1-TN0)
      auc += ((tot._1 - cs._1) + (tot._1 - cs._1 - x._1)) / 2.0 * x._2
      // (TP0, TN0) -> (TP1, TN1)
      cs += x
    }
    auc / tot._1 / tot._2
  }

  // Uses trapezium rule to compute the precision recall AUC.
  private def getPRAUC(pr: mutable.Buffer[(Double, Double, Double)]): Double = {
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

  private def evaluateRecordForAUC(record: EvaluationRecord,
                                   minScore: Double,
                                   maxScore: Double): (Double, (Long, Long, Long, Long)) = {
    var offset = if (record.is_training) 0 else 2
    if (record.label <= 0) {
      offset += 1
    }

    val score: Double = (record.score - minScore) / (maxScore - minScore) * 100

    offset match {
      case 0 => (score, (1, 0, 0, 0))
      case 1 => (score, (0, 1, 0, 0))
      case 2 => (score, (0, 0, 1, 0))
      case 3 => (score, (0, 0, 0, 1))
    }
  }

  private def evaluateRecordRegression(record: EvaluationRecord): Iterator[(String, Double)] = {
    val out = collection.mutable.ArrayBuffer[(String, Double)]()

    val prefix = if (record.is_training) "TRAIN_" else "HOLD_"
    val diff = record.label - record.score
    val sqErr = diff * diff
    // to compute SMAPE third version https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    val absLabelMinusScore = Math.abs(diff)
    val labelPlusScore = record.label + record.score
    out.append((prefix + "SQERR", sqErr))
    out.append((prefix + "ABS_LABEL_MINUS_SCORE", absLabelMinusScore))
    out.append((prefix + "LABEL_PLUS_SCORE", labelPlusScore))
    out.append((prefix + "COUNT", 1.0))
    out.iterator
  }

  // Given a record and a threshold compute if it is a true/false positive/negative
  // and the squared error assuming it is a probability.
  private def evaluateRecordBinaryClassification(threshold: Double)
                                                (record: EvaluationRecord): Iterator[(String, Double)] = {
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

  private def getThresholds(minScore: Double, maxScore: Double, bucketCount: Int): Array[Double] = {
    // ref: https://github.com/apache/spark/blob/master/core/src/main/scala/org/apache/spark/rdd/DoubleRDDFunctions.scala
    def customRange(min: Double, max: Double, steps: Int): IndexedSeq[Double] = {
      val span = max - min
      Range.Int(0, steps, 1).map(s => min + (s * span) / steps) :+ max
    }
    val range = if (minScore != maxScore) {
      // Range.Double.inclusive(min, max, increment)
      // The above code doesn't always work. See Scala bug #SI-8782.
      // https://issues.scala-lang.org/browse/SI-8782
      customRange(minScore, maxScore, bucketCount)
    } else {
      List(minScore, minScore)
    }
    range.toArray
  }
}
