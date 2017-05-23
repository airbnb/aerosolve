package com.airbnb.aerosolve.training.strategy.eval

import scala.util.Try


case class BinaryMetrics(posCount: Int,
                         negCount: Int,
                         posSugHigher: Int,
                         posSugLower: Int,
                         negSugHigher: Int,
                         negSugLower: Int,
                         increasePrecision: Double,
                         increaseRecall: Double,
                         decreasePrecision: Double,
                         decreaseRecall: Double,
                         trueRegret: Double,
                         trueRegretMedian: Double,
                         trueRegret75Percentile: Double,
                         falseRegret: Double,
                         trueIncreaseMagnitude: Double,
                         trueDecreaseMagnitude: Double,
                         falseDecreaseMagnitude: Double,
                         falseIncreaseMagnitude: Double) {
  def toTSVRow: String = {
    Vector(
      posCount, negCount, posSugHigher, posSugLower, negSugHigher, negSugLower, // raw counts
      increasePrecision, increaseRecall, decreasePrecision, decreaseRecall, // precision-recall metrics
      trueRegret, trueRegretMedian, trueRegret75Percentile, falseRegret,
      trueIncreaseMagnitude,
      trueDecreaseMagnitude,
      falseDecreaseMagnitude,
      falseIncreaseMagnitude // magnitude metrics
    ).mkString("\t")
  }

  // For ease of printing with field names
  def toArray: Array[(String, Any)] = {
    this.getClass
      .getDeclaredFields
      .map( _.getName ) // all field names
      .zip( this.productIterator.to )
  }
}

object BinaryMetrics {
  val evalMetrics = Vector(
    "posCount", "negCount", "posSugHigher", "posSugLower", "negSugHigher", "negSugLower",
    "increasePrecision", "increaseRecall", "decreasePrecision", "decreaseRecall",
    "trueRegret", "trueRegretMedian", "trueRegret75Percentile", "falseRegret",
    "trueIncreaseMagnitude",
    "trueDecreaseMagnitude",
    "falseDecreaseMagnitude",
    "falseIncreaseMagnitude"
  )

  val evalMetricsHeader: String = evalMetrics.mkString("\t")

  def computeEvalMetricFromCounts(results: Map[(Boolean, Boolean), (Int, Double)],
                                  trueRegretMedian: Double,
                                  trueRegret75Percentile: Double
                                 ): BinaryMetrics = {
    val posSugHigher:Int = Try(results(true, false)._1).getOrElse(0)
    val posSugLower:Int = Try(results(true, true)._1).getOrElse(0)
    val negSugHigher:Int = Try(results(false, false)._1).getOrElse(0)
    val negSugLower:Int = Try(results(false, true)._1).getOrElse(0)

    val posCount = posSugHigher + posSugLower
    val negCount = negSugHigher + negSugLower
    val lowerCount = posSugLower + negSugLower
    val higherCount = posSugHigher + negSugHigher

    val trueDecreaseSum = Try(results(true, true)._2).getOrElse(0.0)
    val trueIncreaseSum = Try(results(true, false)._2).getOrElse(0.0)
    val falseDecreaseSum = Try(results(false, true)._2).getOrElse(0.0)
    val falseIncreaseSum = Try(results(false, false)._2).getOrElse(0.0)

    BinaryMetrics(
      posCount = posCount,
      negCount = negCount,
      posSugHigher = posSugHigher,
      posSugLower = posSugLower,
      negSugHigher = negSugHigher,
      negSugLower = negSugLower,
      increasePrecision = 1.0 * posSugHigher / higherCount,
      increaseRecall = 1.0 * posSugHigher / posCount,
      decreasePrecision = 1.0 * negSugLower / lowerCount,
      decreaseRecall = 1.0 * negSugLower / negCount,
      trueRegret = trueDecreaseSum / posCount,
      trueRegretMedian = trueRegretMedian,
      trueRegret75Percentile = trueRegret75Percentile,
      falseRegret = falseIncreaseSum / negCount,
      trueIncreaseMagnitude = safeDiv(trueIncreaseSum, posSugHigher),
      trueDecreaseMagnitude = safeDiv(trueDecreaseSum, posSugLower),
      falseDecreaseMagnitude = safeDiv(falseDecreaseSum, negSugLower),
      falseIncreaseMagnitude = safeDiv(falseIncreaseSum, negSugHigher)
    )
  }

  def safeDiv(numerator: Double, denominator: Long): Double = {
    if (denominator == 0) {
      0
    } else {
      numerator / denominator
    }
  }

}

