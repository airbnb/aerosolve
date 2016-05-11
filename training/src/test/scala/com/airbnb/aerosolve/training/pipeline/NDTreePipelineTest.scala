package com.airbnb.aerosolve.training.pipeline

import java.util

import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.core.models.AdditiveModel
import com.airbnb.aerosolve.training.TrainingTestHelper
import org.junit.Assert._
import org.junit.Test

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class NDTreePipelineTest {
  @Test
  def examplesToFloatFeatureArray() = {
    val example: Example = TrainingTestHelper.makeExample(1, 2, 3)
    val map =  mutable.Map[(String, String), ArrayBuffer[Array[Double]]]()
    val emptyLinearFamilies: java.util.List[String] = List.empty.asJava
    NDTreePipeline.examplesToFloatFeatureArray(example, emptyLinearFamilies, map)
    assertEquals(3, map.size)
    val b1: ArrayBuffer[Array[Double]] = map.get(("loc", "x")).get
    assertEquals(1, b1(0)(0), 0)
    val b2: ArrayBuffer[Array[Double]] = map.get(("loc", "y")).get
    assertEquals(2, b2(0)(0), 0)

    val b3: ArrayBuffer[Array[Double]] = map.get(("$rank", "")).get
    assertEquals(3, b3(0)(0), 0)

    val linearFamilies: java.util.List[String] = util.Arrays.asList("loc")
    val map2 =  mutable.Map[(String, String), ArrayBuffer[Array[Double]]]()
    NDTreePipeline.examplesToFloatFeatureArray(example, linearFamilies, map2)
    assertEquals(1, map2.size)
    val b4: ArrayBuffer[Array[Double]] = map2.get(("$rank", "")).get
    assertEquals(3, b4(0)(0), 0)
  }

  @Test
  def examplesToDenseFeatureArray() = {
    val example: Example = TrainingTestHelper.makeDenseExample(1, 2, 3)
    val map =  mutable.Map[(String, String), ArrayBuffer[Array[Double]]]()
    NDTreePipeline.examplesToDenseFeatureArray(example, map)
    assertEquals(1, map.size)
    val b1: ArrayBuffer[Array[Double]] = map.get((AdditiveModel.DENSE_FAMILY, "d")).get
    assertEquals(1, b1(0)(0), 0)
    assertEquals(2, b1(0)(1), 0)
  }
}
