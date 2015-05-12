package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.training.CyclicCoordinateDescent.Params
import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.core.{Example, FeatureVector}
import com.typesafe.config.{ConfigFactory, Config}
import org.apache.spark.SparkContext
import org.junit.Test
import org.slf4j.LoggerFactory
import org.junit.Assert.assertEquals

import scala.collection.mutable.ArrayBuffer

class CyclicCoordinateDescentTest {
  val log = LoggerFactory.getLogger("CyclicCoordinateDescentTest")

  @Test def quadraticTest: Unit = {
    val iterations = 5
    val initial = Array(0.0, 0.0)
    val initialStep = Array(1.0, 1.0)
    val bounds : Array[(Double, Double)] = Array((-10.0, 10.0), (-10.0, 10.0))
    val params = Params(0.1, iterations, initial, initialStep, bounds)
    def f(x : Array[Double]) = {
      (x(0) + 5.0) * (x(0) - 3.0) +
      (x(1) + 2.0) * (x(1) - 7.0)
    }
    val best = CyclicCoordinateDescent.optimize(f, params)
    assertEquals(-1.0, best(0), 0.1)
    assertEquals(2.5, best(1), 0.1)
  }
}