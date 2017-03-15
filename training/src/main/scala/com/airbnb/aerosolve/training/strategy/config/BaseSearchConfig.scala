package com.airbnb.aerosolve.training.strategy.config

import com.typesafe.config.Config
import org.slf4j.{Logger, LoggerFactory}

case class BaseSearchConfig(searchParams: SearchParams[Double],
                            table: String,
                            partition: String,
                            trainingOptions: TrainingOptions
                           ) {
  def getTrainingOptions: Array[TrainingOptions] = {
    searchParams.paramCombinations.map(currentParams =>
      trainingOptions.updateTrainingOptions(searchParams.paramNames, currentParams)
    ).toArray
  }
}

object BaseSearchConfig {
  val log: Logger = LoggerFactory.getLogger("BaseSearchConfig")
  def loadConfig(config: Config): BaseSearchConfig = {
    val taskConfig = config.getConfig("param_search")
    val searchParams = SearchParams.loadDoubleFromConfig(taskConfig)
    val table = taskConfig.getString("table")
    val partition = taskConfig.getString("partition")
    val trainingOptions = TrainingOptions.loadBaseTrainingOptions(
      config.getConfig("training_options"))

    BaseSearchConfig(
      searchParams,
      table,
      partition,
      trainingOptions)
  }
}
