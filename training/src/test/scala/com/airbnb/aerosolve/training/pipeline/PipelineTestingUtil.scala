package com.airbnb.aerosolve.training.pipeline

import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.mockito.Matchers._
import org.mockito.Mockito._

import scala.language.implicitConversions
import scala.reflect.ClassTag
import scala.reflect.runtime.universe.TypeTag

/*
 * Misc. utilities that may be useful for testing Spark pipelines.
 */
object PipelineTestingUtil {
  def generateSparkContext = {
    val sparkConf =
      new SparkConf()
        .setMaster("local[2]")
        .setAppName("PipelineTestingUtil")
        .set("spark.io.compression.codec", "lz4")

    new SparkContext(sparkConf)
  }

  /*
   * Wrapper that generates a local SparkContext for a test and then
   * cleans everything up after test completion.
   */
  def withSparkContext[B](f: SparkContext => B): B = {
    val sc = generateSparkContext

    try {
      f(sc)
    } finally {
      sc.stop
      System.clearProperty("spark.master.port")
    }
  }

  /*
   * Create a mock HiveContext that responds to sql calls by returning
   * the argument results in a SchemaRDD.
   *
   * Works for any case class.
   */
  def createFakeHiveContext[A <: Product: TypeTag : ClassTag](
      sc: SparkContext, results: Seq[A]): HiveContext = {
    val mockHiveContext = mock(classOf[HiveContext])

    val sqlContext = new SQLContext(sc)

    when(mockHiveContext.sql(anyString())).thenReturn(sqlContext.createDataFrame(results))

    mockHiveContext
  }

  /*
   * Create a fake row and schema for a given case class.
   */
  def createFakeRowAndSchema[A <: Product: TypeTag : ClassTag](
      sc: SparkContext, result: A): (Row, StructType) = {
    val sqlContext = new SQLContext(sc)

    val sqlResult = sqlContext.createDataFrame(Seq(result))

    (sqlResult.head(), sqlResult.schema)
  }
}
