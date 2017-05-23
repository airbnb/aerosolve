package com.airbnb.aerosolve.training.strategy.data

trait BinaryScoringSample {
  def observedValue: Double

  // the value used to prediction
  // can be avg(observed value) depends on implementation
  def pivotValue: Double

  def x: Double

  // TODO maybe need predictionHigher?
  def predictionLower(prediction: Double): Boolean = {
    prediction < observedValue
  }

  def predictionIncrease(prediction: Double): Double = {
    prediction - observedValue
  }
}
