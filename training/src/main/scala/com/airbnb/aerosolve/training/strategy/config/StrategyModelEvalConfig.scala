package com.airbnb.aerosolve.training.strategy.config

import com.airbnb.aerosolve.training.strategy.data.TrainingData
import com.typesafe.config.Config
import org.apache.spark.sql.hive.HiveContext
import org.slf4j.{Logger, LoggerFactory}

import scala.language.existentials
import scala.util.Try

case class StrategyModelEvalConfig(trainingDataQuery: String,
                                   evalDataQuery: String,
                                   partitions: Int,
                                   shuffle: Boolean) {
}

object DirectQueryEvalConfig {
  val log: Logger = LoggerFactory.getLogger(this.getClass.getName)

  def loadConfig[T](hc: HiveContext,
                 config: Config,
                 data: TrainingData[T]): StrategyModelEvalConfig = {
    val evalConfig = config.getConfig("param_search")
    val query = Try(evalConfig.getBoolean("direct_query")).getOrElse(false)

    val trainingDataQuery = config.getString("training_data_query")
    val evalDataQuery = config.getString("eval_data_query")
    val partitions = Try(evalConfig.getInt("partition_num")).getOrElse(5000)
    val shuffle = Try(evalConfig.getBoolean("shuffle")).getOrElse(false)

    log.info(s"Training Data Query: $trainingDataQuery")
    log.info(s"Eval Data Query: $evalDataQuery")

    StrategyModelEvalConfig(
      trainingDataQuery,
      evalDataQuery,
      partitions,
      shuffle)
  }
}

