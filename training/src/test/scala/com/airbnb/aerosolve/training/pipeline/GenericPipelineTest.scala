package com.airbnb.aerosolve.training.pipeline

import com.airbnb.aerosolve.core.features.FeatureRegistry
import com.google.common.collect.ImmutableMap
import org.junit.Assert._
import org.junit.Test

import scala.collection.JavaConverters._

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
  val registry = new FeatureRegistry

  @Test
  def hiveTrainingToExample() = {
    val fakeRow = FakeDataRow(
      10, 7, 4.1f, 11.0, false, "some string", "some other string", 4.5
    )

    PipelineTestingUtil.withSparkContext(sc => {
      val (sqlRow, schema) = PipelineTestingUtil.createFakeRowAndSchema(sc, fakeRow)

      val example = GenericPipeline.hiveTrainingToExample(sqlRow, schema.fields.toArray, registry,
                                                          false)
      val fv = example.only

      assertTrue(fv.containsKey("s", "stringFeature:some string"))
      assertTrue(fv.containsKey("s2", "some other string"))

      assertEquals(fv.get("i", "intFeature1"), 10.0, 0.01)
      assertEquals(fv.get("i", "intFeature2"), 7.0, 0.01)
      assertEquals(fv.get("f", "floatFeature"), 4.1, 0.01)
      assertEquals(fv.get("d", "sparseFeature"), 11.0, 0.01)

      assertTrue(fv.containsKey("b", "boolFeature:F"))

      assertEquals(fv.get("LABEL", ""), 4.5, 0.01)
    })
  }

  @Test
  def hiveTrainingToExampleMulticlass() = {
    val fakeRow = FakeDataRowMulticlass(
      10, "CLASS1:2.1,CLASS2:4.5,CLASS3:5.6"
    )

    PipelineTestingUtil.withSparkContext(sc => {
      val (sqlRow, schema) = PipelineTestingUtil.createFakeRowAndSchema(sc, fakeRow)

      val example = GenericPipeline.hiveTrainingToExample(sqlRow, schema.fields.toArray, registry,
                                                          true)
      val fv = example.only

      assertEquals(fv.get("i", "intFeature"), 10.0, 0.01)

      val labelMap = fv.get(registry.family("label"))
        .iterator.asScala.map(fv => (fv.feature.name, fv.value))
        .toMap

      assertEquals(
        labelMap,
        ImmutableMap.of(
          "CLASS1", 2.1,
          "CLASS2", 4.5,
          "CLASS3", 5.6
        )
      )
    })
  }
}
