package com.airbnb.aerosolve.training.utils

import org.apache.spark.SparkContext
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.hive.test.TestHiveContext
import org.junit.{After, Before}

class TestWithHiveContext {
  var sc : SparkContext = _
  var hc : HiveContext = _

  @Before
  def init(): Unit = {
    this.sc = new SparkContext("local", this.getClass.toString)
    this.hc = new TestHiveContext(sc)
  }

  @After
  def cleanup(): Unit = {
    sc.stop()
    System.clearProperty("spark.master.port")
  }
}
