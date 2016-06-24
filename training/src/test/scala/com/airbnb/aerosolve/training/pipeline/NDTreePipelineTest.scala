package com.airbnb.aerosolve.training.pipeline

import java.util

import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.core.models.AdditiveModel
import com.airbnb.aerosolve.training.TrainingTestHelper
import com.airbnb.aerosolve.training.pipeline.NDTreePipeline.{FeatureStats, NDTreePipelineParams}
import org.junit.Assert._
import org.junit.Test

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class NDTreePipelineTest {
  @Test
  def examplesToFloatFeatureArray() = {
    val example: Example = TrainingTestHelper.makeExample(1, 2, 3)
    val map = mutable.Map[(String, String), Either[ArrayBuffer[Array[Double]], FeatureStats]]()
    val emptyLinearFamilies: java.util.List[String] = List.empty.asJava
    val emptySplineFamilies: java.util.List[String] = List.empty.asJava
    val params = NDTreePipelineParams(
      0, 0, emptyLinearFamilies, emptySplineFamilies, "", 0, 0, 0, 0, 0)
    NDTreePipeline.examplesToFloatFeatureArray(example, params, map)
    assertEquals(3, map.size)
    val b1: ArrayBuffer[Array[Double]] = map.get(("loc", "x")).get.left.get
    assertEquals(1, b1(0)(0), 0)
    val b2: ArrayBuffer[Array[Double]] = map.get(("loc", "y")).get.left.get
    assertEquals(2, b2(0)(0), 0)

    val b3: ArrayBuffer[Array[Double]] = map.get(("$rank", "")).get.left.get
    assertEquals(3, b3(0)(0), 0)

    val linearFamilies: java.util.List[String] = util.Arrays.asList("loc")
    val map2 = mutable.Map[(String, String), Either[ArrayBuffer[Array[Double]], FeatureStats]]()
    val paramsWithLinearFamilies = NDTreePipelineParams(
      0, 0, linearFamilies, emptySplineFamilies, "", 0, 0, 0, 0, 0)
    NDTreePipeline.examplesToFloatFeatureArray(example, paramsWithLinearFamilies, map2)
    // linear feature return FeatureStats
    assertEquals(3, map2.size)
    val b5: FeatureStats = map2.get(("loc", "x")).get.right.get
    assertEquals(1, b5.count, 0)
    assertEquals(1, b5.max, 0)
    assertEquals(1, b5.min, 0)
    assertFalse(b5.spline)
    val b6: FeatureStats = map2.get(("loc", "y")).get.right.get
    assertEquals(1, b6.count, 0)
    assertEquals(2, b6.max, 0)
    assertEquals(2, b6.min, 0)
    assertFalse(b6.spline)

    val b4: ArrayBuffer[Array[Double]] = map2.get(("$rank", "")).get.left.get
    assertEquals(3, b4(0)(0), 0)

    val splineFamilies: java.util.List[String] = util.Arrays.asList("loc")
    val map3 = mutable.Map[(String, String), Either[ArrayBuffer[Array[Double]], FeatureStats]]()
    val paramsWithSplineFamilies = NDTreePipelineParams(
      0, 0, emptyLinearFamilies, splineFamilies, "", 0, 0, 0, 0, 0)
    NDTreePipeline.examplesToFloatFeatureArray(example, paramsWithSplineFamilies, map3)
    // linear feature return FeatureStats
    assertEquals(3, map3.size)
    val b7: FeatureStats = map3.get(("loc", "x")).get.right.get
    assertEquals(1, b7.count, 0)
    assertEquals(1, b7.max, 0)
    assertEquals(1, b7.min, 0)
    assertTrue(b7.spline)

    val b8: FeatureStats = map3.get(("loc", "y")).get.right.get
    assertEquals(1, b8.count, 0)
    assertEquals(2, b8.max, 0)
    assertEquals(2, b8.min, 0)
    assertTrue(b8.spline)

    val b9: ArrayBuffer[Array[Double]] = map3.get(("$rank", "")).get.left.get
    assertEquals(3, b9(0)(0), 0)

  }

  @Test
  def examplesToDenseFeatureArray() = {
    val example: Example = TrainingTestHelper.makeDenseExample(1, 2, 3)
    val map = mutable.Map[(String, String), Either[ArrayBuffer[Array[Double]], FeatureStats]]()
    NDTreePipeline.examplesToDenseFeatureArray(example, map)
    assertEquals(1, map.size)
    val b1: ArrayBuffer[Array[Double]] = map.get((AdditiveModel.DENSE_FAMILY, "d")).get.left.get
    assertEquals(1, b1(0)(0), 0)
    assertEquals(2, b1(0)(1), 0)
  }

  @Test def examplesToStringFeatureArray() = {
    val example: Example = TrainingTestHelper.makeExample(-1, -2, 3)
    val map = mutable.Map[(String, String), Either[ArrayBuffer[Array[Double]], FeatureStats]]()
    NDTreePipeline.examplesToStringFeatureArray(example, map)
    assertEquals(2, map.size)
    val b1: FeatureStats = map.get(("BIAS", "B")).get.right.get
    assertEquals(1, b1.count, 0)
    assertEquals(1, b1.min, 0)
  }
}
