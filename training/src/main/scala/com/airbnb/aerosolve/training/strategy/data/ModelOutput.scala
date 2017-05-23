package com.airbnb.aerosolve.training.strategy.data

import com.airbnb.aerosolve.training.strategy.config.TrainingOptions
import com.airbnb.aerosolve.training.strategy.eval.BinaryMetrics
import com.airbnb.aerosolve.training.strategy.params.StrategyParams
import com.airbnb.aerosolve.training.utils.HiveManageTable
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SaveMode}

case class ModelOutput[T](id: String,
                          params: StrategyParams[T],
                          loss: Double,
                          metrics: BinaryMetrics,
                          options: TrainingOptions) extends HiveManageTable{
  override def toRow(partition: String): Row = {
    Row(
      id.toLong,
      metrics.posCount,
      metrics.negCount,
      metrics.posSugHigher,
      metrics.posSugLower,
      metrics.negSugHigher,
      metrics.negSugLower,
      metrics.increasePrecision,
      metrics.increaseRecall,
      metrics.decreasePrecision,
      metrics.decreaseRecall,
      metrics.trueRegret,
      metrics.trueRegretMedian,
      metrics.trueRegret75Percentile,
      metrics.falseRegret,
      metrics.trueIncreaseMagnitude,
      metrics.trueDecreaseMagnitude,
      metrics.falseDecreaseMagnitude,
      metrics.falseIncreaseMagnitude,
      params.params,
      loss,
      options.toPartialArray,
      partition
    )
  }
}

object ModelOutput {
  lazy val schema = StructType(
    Seq(
      StructField("id", LongType),
      StructField("posCount", IntegerType),
      StructField("negCount", IntegerType),
      StructField("posSugHigher", IntegerType),
      StructField("posSugLower", IntegerType),
      StructField("negSugHigher", IntegerType),
      StructField("negSugLower", IntegerType),

      StructField("increasePrecision", DoubleType),
      StructField("increaseRecall", DoubleType),
      StructField("decreasePrecision", DoubleType),
      StructField("decreaseRecall", DoubleType),

      StructField("trueRegret", DoubleType),
      StructField("trueRegretMedian", DoubleType),
      StructField("trueRegret75Percentile", DoubleType),

      StructField("falseRegret", DoubleType),
      StructField("trueIncreaseMagnitude", DoubleType),
      StructField("trueDecreaseMagnitude", DoubleType),
      StructField("falseDecreaseMagnitude", DoubleType),
      StructField("falseIncreaseMagnitude", DoubleType),

      StructField("params", ArrayType(DoubleType)),
      StructField("loss", DoubleType),
      StructField("options", ArrayType(DoubleType)),

      StructField("model", StringType)
    )
  )

  def save[T](hiveContext: HiveContext,
              data: RDD[ModelOutput[T]],
              table: String,
              partition: String): Unit = {
    HiveManageTable.saveRDDToHive(
      hiveContext,
      data,
      table,
      ModelOutput.schema,
      SaveMode.Overwrite,
      "model",
      partition)
  }
}
