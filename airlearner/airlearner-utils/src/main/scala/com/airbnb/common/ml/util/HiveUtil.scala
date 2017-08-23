package com.airbnb.common.ml.util

import scala.reflect.ClassTag
import scala.util.Try

import com.typesafe.config.Config
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.{DataFrame, Row}

object HiveUtil {

  /*
   * Drop the specified Hive partition if it exists
   */
  def dropHivePartition(
      hc: HiveContext,
      hiveTableName: String,
      hivePartitionSpecs: Map[String, Any]
  ): Unit = {
    if (!hiveTableName.contains('.')) {
      throw new RuntimeException(s"Missing namespace for hive table: $hiveTableName.")
    }

    // Break the table name into namespace.table_name
    val Array(namespace, table) = hiveTableName.split('.')
    // Turn the partition spec map into a Hive-format String
    val partitionSpec = hivePartitionSpecsMapToString(hivePartitionSpecs)

    // Drop the partition if it exists
    hc.sql(s"USE $namespace")
    hc.sql(s"ALTER TABLE $table DROP IF EXISTS PARTITION ($partitionSpec)")
  }

  /*
   * Drop the specified Hive partition if it exists
   */
  def dropHivePartition(
      hc: HiveContext,
      hiveTableName: String,
      hivePartitionSpecs: String
  ): Unit = {
    if (!hiveTableName.contains('.')) {
      throw new RuntimeException(s"Missing namespace for hive table: $hiveTableName.")
    }

    // Break the table name into namespace.table_name
    val Array(namespace, table) = hiveTableName.split('.')

    // Drop the partition if it exists
    hc.sql(s"USE $namespace")
    hc.sql(s"ALTER TABLE $table DROP IF EXISTS PARTITION ($hivePartitionSpecs)")
  }

  def updateHivePartition(
      hc: HiveContext,
      hiveTable: String,
      partitionSpec: String,
      hdfsLocation: String
  ): Boolean = {
    if (!hiveTable.contains('.')) {
      throw new RuntimeException(s"Missing namespace for the hive table $hiveTable.")
    }
    val Array(namespace, table) = hiveTable.split('.')
    hc.sql(s"USE $namespace")
    hc.sql(s"ALTER TABLE $table DROP IF EXISTS PARTITION ($partitionSpec)")
    hc.sql(s"ALTER TABLE $table ADD PARTITION ($partitionSpec) location '$hdfsLocation'")
    true
  }

  def loadDataFromHive[T](
      hiveContext: HiveContext,
      dataQuery: String,
      parseKeyFromHiveRow: (Row) => String,
      parseSampleFromHiveRow: (Row) => T
  )(implicit c: ClassTag[T]):
  RDD[(String, T)] = {
    loadDataFromDataFrame(hiveContext.sql(dataQuery), parseKeyFromHiveRow, parseSampleFromHiveRow)
  }

  def loadDataFromDataFrame[T](
      data: DataFrame,
      parseKeyFromHiveRow: (Row) => String,
      parseSampleFromHiveRow: (Row) => T
  )
    (implicit c: ClassTag[T]): RDD[(String, T)] = {
    data.map(row => {
      val key = parseKeyFromHiveRow(row)
      val t = parseSampleFromHiveRow(row)
      (key, t)
    })
  }

  def loadDataFromDataFrameGroupByKey[T: ClassTag](
      data: DataFrame,
      parseKeyFromHiveRow: (Row) => String,
      parseSampleFromHiveRow: (Row) => T
  ): RDD[(String, Seq[T])] = {
    data.map(row => {
      val key = parseKeyFromHiveRow(row)
      val t = parseSampleFromHiveRow(row)
      (key, t)
    })
      .groupByKey
      .mapValues(sample => sample.toSeq)
  }

  def parseLongToStringFromHiveRow(key: String)(row: Row): String = {
    row.getAs[Long](key).toString
  }

  def parseDoubleArrayFromHiveRow(field: String)(row: Row): Array[Double] = {
    row.getAs[scala.collection.mutable.WrappedArray[Double]](field).toArray
  }

  //
  // default always overwrite
  /**
    * Save result to output as textfile and update outputTable as partitionSpec.
    * By default this overwrites (unless `$.overwrite = false`).
    *
    * @param hc     HiveContext
    * @param config top-level config instance
    * @param result data to save to Hive
    */
  def saveToHiveWithConfig[U](
      hc: HiveContext,
      config: Config,
      result: RDD[U]
  ): Unit = {
    // Get the output path from the config file
    val output: String = config.getString("output_path")
    val outputTable: String = config.getString("output_table")

    // update hive partition
    val partitionSpec: String = config.getString("output_partition_spec")
    val overwrite: Boolean = Try(config.getBoolean("overwrite")).getOrElse(true)

    // This decouples computing and storage parallelism of this pipeline:
    // The final params are incredibly small, about 50MB in total for all listings
    //
    // However we would like to have a large parallelism for training, where 500 partitions will
    // take up to 3 hours to train. `spark.default.parallelism` continues to control this
    // parallelism and by having a high parallelism, we will also do better in case of data
    // imbalance.
    //
    // NB: Use repartition instead of coalesce here because Spark would coalesce early and use final
    // parallelism during taining stage. Again the output is small enough that shuffle does not
    // introduce much overhead.
    val outputPartitions: Int = Try(config.getInt("output_partitions")).getOrElse(5)

    PipelineUtil.saveToHdfsAndUpdateHive(
      hc,
      output,
      outputTable,
      partitionSpec,
      result.repartition(outputPartitions),
      overwrite)

    // save to a different schema for online systme
    // i.e. HFile requires a different schema with same HDFS file
    // update hive partitions by output_table_hfile
    // todo remove hfile from output_table_hfile
    if (config.hasPath("output_table_hfile")) {
      val hfileTable: String = config.getString("output_table_hfile")
      HiveUtil.updateHivePartition(hc, hfileTable, partitionSpec, output)
    }
  }

  /*
   * Convert a partition_specs map to a usable Hive string.
   * ex:
   *   Map("ds" -> "2016-11-12", "night" -> 3)
   *   =>
   *   "ds='2016-11-12', night=3"
   */
  private def hivePartitionSpecsMapToString(hivePartitionSpecs: Map[String, Any]): String = {
    hivePartitionSpecs
      .map {
        // If the value is string type, surround it by single-quotes
        case (column, value: String) => s"$column='$value'"
        case (column, value) => s"$column=$value"
      }
      .mkString(", ")
  }
}
