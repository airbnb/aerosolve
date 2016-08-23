package com.airbnb.aerosolve.training.photon.ml.data

import com.airbnb.aerosolve.core.features.Features
import com.airbnb.aerosolve.core.Example
import org.junit.Assert._
import org.junit.Test

import scala.collection.immutable

/**
  * This class tests [[SingleFloatFamilyMetaExtractor]].
  *
  */
class SingleFloatFamilyMetaExtractorTest {

  private def mockExampleMetaFields(metaFamily: String): Example = {
    val builder = Features.builder()
    builder.names(Array[String](metaFamily + "_foo",
      metaFamily + "_bar",
      metaFamily + "_tar",
      "randomF_b"))
    builder.values(Array[Object](
      new java.lang.Integer(123),
      new java.lang.Double(100d),
      new java.lang.Long(-321L),
      new java.lang.Integer(20)))
    builder.build().toExample(false)
  }

  @Test
  def testMetaDataExtraction(): Unit = {
    val extractor = new SingleFloatFamilyMetaExtractor()
    val example = mockExampleMetaFields("meta")
    val metaMap = extractor.buildMetaDataMap(example, immutable.Map[String, String]())

    assertEquals(3, metaMap.size())
    assertEquals(123L, metaMap.get("foo"))
    assertEquals(100L, metaMap.get("bar"))
    // Should make all ids >= 0
    assertEquals(321L, metaMap.get("tar"))

    val example2 = mockExampleMetaFields("metav2")
    val metaMap2 = extractor.buildMetaDataMap(example2, immutable.Map[String, String]("metaFamily" -> "metav2"))
    assertEquals(3, metaMap2.size())
    assertEquals(123L, metaMap2.get("foo"))
    assertEquals(100L, metaMap2.get("bar"))
    assertEquals(321L, metaMap2.get("tar"))
  }

  @Test(expected = classOf[IllegalArgumentException])
  def testMissingField(): Unit = {
    val extractor = new SingleFloatFamilyMetaExtractor()
    val example = mockExampleMetaFields("metav2")
    extractor.buildMetaDataMap(example, immutable.Map[String, String]())
  }
}
