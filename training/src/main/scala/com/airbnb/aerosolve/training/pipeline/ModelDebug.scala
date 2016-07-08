package com.airbnb.aerosolve.training.pipeline

import com.airbnb.aerosolve.core.ModelRecord
import com.airbnb.aerosolve.core.function.{AbstractFunction, MultiDimensionSpline}
import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.training.pipeline.HiveUtil
import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.sql.hive.HiveContext
import org.slf4j.{Logger, LoggerFactory}

import scala.util.Try

/*
  Once you create table like below
  CREATE EXTERNAL TABLE ndtree_dump(
    feature_family STRING,
    feature_name STRING,
    min_value FLOAT,
    max_value FLOAT,
    knot_values  STRING,
    ndtree_node STRING,
    min_leaf_nodes_count_div_by_std FLOAT,
    tolerance FLOAT,
    tolerancePercentage FLOAT,
    )
  PARTITIONED BY (
    model  STRING)
  ROW FORMAT DELIMITED FIELDS TERMINATED BY '\001'
  STORED AS TEXTFILE LOCATION
  'hdfs://hdfs/team/debug/ndtree_model'
  then you run DumpModelToHive to generate the partition
  and add it by ALTER TABLE xxx ADD PARTITION () LOCATION ''
  sample config for DumpModelToHive
  dump_model_to_hive {
    model_name : "hdfs://hdfs/team/debug/ndtree_model"
    model_dump : "hdfs://hdfs/team//model_coefficient_spline_values/model=xxx"
    overwrite : true
  }
*/
object ModelDebug {
  val log: Logger = LoggerFactory.getLogger("DebugPipeline")
  def modelRecordToString(x: ModelRecord) : String = {
    if (x.weightVector != null && !x.weightVector.isEmpty) {
      val func = AbstractFunction.buildFunction(x)
      val tolerance = func.smooth(0, false)
      val tolerancePercentage = func.smooth(0, true)
      val nDTreeModelString: String = if (x.ndtreeModel != null) {
        val w = func.asInstanceOf[MultiDimensionSpline].getWeightsString
        x.ndtreeModel.toString + w

      } else {
        ""
      }

      s"%s\u0001%s\u0001%f\u0001%f\u0001%s\u0001%s\u0001%f\u0001%f\u0001%f".format(
        x.featureFamily, x.featureName, x.minVal, x.maxVal, x.weightVector.toString,
        nDTreeModelString, 0.0, tolerance, tolerancePercentage)
    } else {
      log.info(s" ${x.featureFamily} ${x.featureName} miss weightVector")
      ""
    }
  }

  def dumpModelForHive(sc: SparkContext, config: Config) = {
    val cfg = config.getConfig("dump_model_to_hive")
    dumpModel(sc, cfg, x => x.featureName != null, modelRecordToString)
  }

  def dumpModel(sc: SparkContext, config: Config,
    filterFunction: (ModelRecord) => Boolean,
    recordToString: (ModelRecord) => String): Unit = {
    val modelName = config.getString("model_name")
    val modelDump = config.getString("model_dump")
    val outputHiveTable = Try(config.getString("output_hive_table")).getOrElse("")

    val overwrite: Boolean = Try(config.getBoolean("overwrite")).getOrElse(false)

    val model = sc
      .textFile(modelName)
      .map(Util.decodeModel)
      .filter(filterFunction)
      .map(recordToString)
      .filter(_.length > 0)

    PipelineUtil.saveAndCommitAsTextFile(model, modelDump, overwrite)

    if (!outputHiveTable.isEmpty) {
      val hiveContext = new HiveContext(sc)
      val partitionKey = config.getString("partition_key")
      val partitionValue = config.getString("partition_value")
      // assume the value of partition key is string
      HiveUtil.updateHivePartition(
        hiveContext, outputHiveTable, s"$partitionKey='$partitionValue'", modelDump
      )
    }
  }
}
