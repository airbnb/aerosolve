package com.airbnb.aerosolve.training.pipeline

import com.google.common.collect.{ImmutableMap, ImmutableSet}
import org.junit.Assert._
import org.junit.Test

case class FakeDataRow(
    i_intFeature1: Int,
    i_intFeature2: Int,
    f_floatFeature: Float,
    d_doubleFeature: Double,
    b_boolFeature: Boolean,
    s_stringFeature: String,
    s2_RAW: String,
    LABEL: Double)

case class FakeDataRowMulticlass(
    i_intFeature: Int,
    LABEL: String)

class GenericPipelineTest {
  @Test
  def hiveTrainingToExample() = {
    val fakeRow = FakeDataRow(
      10, 7, 4.1f, 11.0, false, "some string", "some other string", 4.5
    )

    PipelineTestingUtil.withSparkContext(sc => {
      val (sqlRow, schema) = PipelineTestingUtil.createFakeRowAndSchema(sc, fakeRow)

      val example = GenericPipeline.hiveTrainingToExample(sqlRow, schema.fields.toArray, false)
      val stringFeatures = example.getExample.get(0).getStringFeatures
      val floatFeatures = example.getExample.get(0).getFloatFeatures

      assertEquals(
        stringFeatures.get("s"),
        ImmutableSet.of("stringFeature:some string")
      )
      assertEquals(
        stringFeatures.get("s2"),
        ImmutableSet.of("some other string")
      )
      assertEquals(
        floatFeatures.get("i"),
        ImmutableMap.of("intFeature1", 10.0, "intFeature2", 7.0)
      )
      assertEquals(
        floatFeatures.get("f"),
        ImmutableMap.of("floatFeature", 4.1f.toDouble)
      )
      assertEquals(
        floatFeatures.get("d"),
        ImmutableMap.of("doubleFeature", 11.0)
      )
      assertEquals(
        stringFeatures.get("b"),
        ImmutableSet.of("boolFeature:F")
      )
      assertEquals(
        floatFeatures.get("LABEL"),
        ImmutableMap.of("", 4.5)
      )
    })
  }

  @Test
  def hiveTrainingToExampleMulticlass() = {
    val fakeRow = FakeDataRowMulticlass(
      10, "CLASS1:2.1,CLASS2:4.5,CLASS3:5.6"
    )

    PipelineTestingUtil.withSparkContext(sc => {
      val (sqlRow, schema) = PipelineTestingUtil.createFakeRowAndSchema(sc, fakeRow)

      val example = GenericPipeline.hiveTrainingToExample(sqlRow, schema.fields.toArray, true)
      val floatFeatures = example.getExample.get(0).getFloatFeatures

      assertEquals(
        floatFeatures.get("i"),
        ImmutableMap.of("intFeature", 10.0)
      )
      assertEquals(
        floatFeatures.get("LABEL"),
        ImmutableMap.of(
          "CLASS1", 2.1,
          "CLASS2", 4.5,
          "CLASS3", 5.6
        )
      )
    })
  }
}
