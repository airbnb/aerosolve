package com.airbnb.aerosolve.training.strategy.data

object BinarySampleTest {
  def get: BaseBinarySample = {
    BaseBinarySample(true, 0.7, 100, 100)
  }

  def getExamples: List[BaseBinarySample] = {
    List(BaseBinarySample(true, 0.7, 100, 100),
      BaseBinarySample(true, 0.6, 100, 100),
      BaseBinarySample(false, 0.5, 90, 100),
      BaseBinarySample(true, 0.3, 85, 100))
  }

  def getExamplesSeq: Seq[BaseBinarySample] = {
    Seq(BaseBinarySample(true, 0.7, 100, 100),
      BaseBinarySample(true, 0.6, 100, 100),
      BaseBinarySample(false, 0.5, 90, 95),
      BaseBinarySample(true, 0.3, 85, 100))
  }
}
