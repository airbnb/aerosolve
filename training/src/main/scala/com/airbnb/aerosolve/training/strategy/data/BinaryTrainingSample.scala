package com.airbnb.aerosolve.training.strategy.data

import com.airbnb.aerosolve.training.strategy.config.TrainingOptions
import org.apache.spark.sql.Row

trait BinaryTrainingSample extends BinaryScoringSample {
  def label: Boolean

  def lossWithPrediction(prediction: Double): Double = {
    if (label) {
      Math.max(0, observedValue - prediction)
    } else {
      Math.max(0, prediction - observedValue)
    }
  }

  def lossRatioWithPrediction(prediction: Double): Double = {
    lossWithPrediction(prediction) / observedValue
  }

  def getTrueOrPivotValue: Double = {
    if (label) {
      observedValue
    } else {
      pivotValue
    }
  }

  def getMinValue: Double = {
    math.min(observedValue, pivotValue)
  }

  def getLowerBound(options: TrainingOptions): Double = {
    if (label) {
      getLowerBoundByScale(
        options.trueLowerBound,
        getTrueOrPivotValue)
    } else {
      getLowerBoundByScale(
        options.falseLowerBound,
        getMinValue)
    }
  }

  def getLowerBoundByScale(bound: Double, bottomLine: Double): Double = {
    val len = (1 - bound) * 2
    val newBound = 1 - len * (1 - x)
    newBound * bottomLine
  }

  def getUpperBound(options: TrainingOptions): Double = {
    if (label) {
      getUpperBoundForTrueLabel(options)
    } else {
      options.falseUpperBound * getMinValue
    }
  }

  def getUpperBoundForTrueLabel(options: TrainingOptions) : Double = {
    // the lower the prob, the smaller the upper bound
    // p = 0, output = 1, p = 1, output > 1, = 1.x
    val len = (options.trueUpperBound - 1) * 2
    val bound = 1 + len * x
    bound * pivotValue
  }
}

case class BaseBinarySample(label: Boolean,
                            x: Double,
                            pivotValue: Double,
                            observedValue: Double) extends BinaryTrainingSample {
}

object BaseBinarySample extends TrainingData[BaseBinarySample] {
  override def parseSampleFromHiveRow(row: Row): BaseBinarySample = ???

  override def selectData: String = ???
}
