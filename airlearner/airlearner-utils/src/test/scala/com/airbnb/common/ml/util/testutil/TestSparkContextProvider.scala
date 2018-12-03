package com.airbnb.common.ml.util.testutil

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

object TestSparkContextProvider {

  /**
    * Create a new SparkContext for testing
    *
    * @return SparkContext
    */
  def createContext(): SparkContext = {
    val spark = SparkSession.builder
      .config(sparkConf())
      .enableHiveSupport()
      .getOrCreate()

    spark.sparkContext
  }

  /**
    * Thoroughly clear a SparkContext between tests.
    *
    * @param sc SparkContext
    */
  def stopContext(sc: SparkContext): Unit = {
    sc.stop()

    System.clearProperty("spark.driver.port")
    System.clearProperty("spark.master.port")
  }

  /**
    * @return a unique app ID for this context/run
    */
  private def appID: String = {
    this.getClass.getName + math.floor(math.random * 10E4).toLong.toString
  }

  /**
    * @return a test-friendly SparkConf instance
    */
  private def sparkConf(): SparkConf = {
    new SparkConf().
      setMaster("local[*]").
      setAppName("test").
      set("spark.ui.enabled", "false").
      set("spark.app.id", appID)
  }
}
