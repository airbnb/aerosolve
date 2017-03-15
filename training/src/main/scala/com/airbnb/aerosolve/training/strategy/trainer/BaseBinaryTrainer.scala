package com.airbnb.aerosolve.training.strategy.trainer

import com.airbnb.aerosolve.training.strategy.config.TrainingOptions
import com.airbnb.aerosolve.training.strategy.data.{BaseBinarySample, TrainingData}
import com.airbnb.aerosolve.training.strategy.params.StrategyParams
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.{DataFrame, Row}

case class BaseBinaryTrainer(strategyParams: StrategyParams[BaseBinarySample],
                             trainingData: TrainingData[BaseBinarySample])
    extends BinaryTrainer[BaseBinarySample]{
  override def getLearningRate(r0: Double,
                               r1: Double,
                               example: BaseBinarySample,
                               options: TrainingOptions): Double = {
    val x = example.x
    val learningRate = if (example.label) {
      r1 * x
    } else {
      1 - x
    }
    r0 * learningRate
  }

  override def createDataFrameFromModelOutput(models: RDD[(String, StrategyParams[BaseBinarySample])], hc: HiveContext): DataFrame = ???

  override def parseKeyFromHiveRow(row: Row): String = ???
}
