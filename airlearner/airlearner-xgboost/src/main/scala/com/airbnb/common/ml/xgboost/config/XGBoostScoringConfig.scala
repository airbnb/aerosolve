package com.airbnb.common.ml.xgboost.config

import com.typesafe.config.Config
import org.apache.spark.SparkContext

import scala.util.Try


case class XGBoostScoringConfig(
    query: String,
    modelBasePath: String,
    outputPath: String,
    overwrite: Boolean,
    partitionSpec: String,
    outputTable: String,
    saveHiveTable: Boolean,
    groupNumber: Int
)

object XGBoostScoringConfig {

  def loadConfig(sc: SparkContext, config: Config): XGBoostScoringConfig = {
    val query = config.getString("scoring_query")
    val modelBasePath = config.getString("model_output")
    val outputPath = config.getString("scoring_output")
    val overwrite = config.getBoolean("overwrite")
    val partitionSpec = config.getString("scoring_partition_spec")
    val outputTable = config.getString("scoring_table")
    val saveHiveTable = Try(config.getBoolean("save_hive_table")).getOrElse(true)
    val groupNumber = config.getInt("group_number")

    XGBoostScoringConfig(
      query,
      modelBasePath,
      outputPath,
      overwrite,
      partitionSpec,
      outputTable,
      saveHiveTable,
      groupNumber)
  }
}
