package com.airbnb.aerosolve.training.pipeline

import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.{Row, SQLContext, StructType}
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
      sc: SparkContext, results: List[A]): HiveContext = {
    val mockHiveContext = mock(classOf[HiveContext])

    val schemaRdd = sc.parallelize(results)

    val sqlContext = new SQLContext(sc)
    import sqlContext._

    schemaRdd.registerAsTable("rows")

    when(mockHiveContext.sql(anyString())).thenReturn(sql("select * from rows"))

    mockHiveContext
  }

  /*
   * Create a fake row and schema for a given case class.
   */
  def createFakeRowAndSchema[A <: Product: TypeTag : ClassTag](
      sc: SparkContext, result: A): Tuple2[Row, StructType] = {
    // TODO: Investigate whether there is an easier way to do this
    val schemaRdd = sc.parallelize(Seq(result))

    val sqlContext = new SQLContext(sc)
    import sqlContext._

    schemaRdd.registerAsTable("rows")

    val sqlResult = sql("select * from rows")

    (sqlResult.collect().toSeq.head, sqlResult.schema)
  }
}
