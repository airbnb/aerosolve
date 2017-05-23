package com.airbnb.aerosolve.training.utils

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.hive.HiveContext

import scala.reflect.ClassTag

object HiveUtil {
  def updateHivePartition(hc: HiveContext,
                          hiveTable: String,
                          partitionSpec: String,
                          hdfsLocation: String) : Boolean = {
    if (!hiveTable.contains('.')) {
      throw new RuntimeException(s"Missing namespace for the hive table $hiveTable.")
    }
    val Array(namespace, table) = hiveTable.split('.')
    hc.sql(s"USE $namespace")
    hc.sql(s"ALTER TABLE $table DROP IF EXISTS PARTITION ($partitionSpec)")
    hc.sql(s"ALTER TABLE $table ADD PARTITION ($partitionSpec) location '$hdfsLocation'")
    true
  }

  def loadDataFromDataFrameGroupByKey[T:ClassTag](data: DataFrame,
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
}