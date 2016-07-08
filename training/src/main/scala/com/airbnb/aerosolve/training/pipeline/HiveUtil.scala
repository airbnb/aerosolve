package com.airbnb.aerosolve.training.pipeline

import org.apache.spark.sql.hive.HiveContext

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
}