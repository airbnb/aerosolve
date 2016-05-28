package com.airbnb.aerosolve.pipeline.transform

import java.{util => ju}

import com.airbnb.aerosolve.core.FeatureVector
import com.airbnb.aerosolve.pipeline.transformers.{Transformer, Transforms, RowTransformer}
import com.airbnb.aerosolve.pipeline.transformers.continuous.Scaler
import com.airbnb.aerosolve.pipeline.transformers.produce.FamilyProducer
import com.airbnb.aerosolve.pipeline.transformers.select.{VectorSelector, FamilySelector}
import com.google.common.collect.ImmutableSet
import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.sql.Row
import org.junit.Assert._
import org.junit.Test
import com.airbnb.aerosolve.pipeline.AerosolveSupport._
import com.airbnb.aerosolve.pipeline.SparkMLSupport._

/**
 *
 */
class FromConfigTest {

  def makeFeatureVector: FeatureVector = {
    val stringFeatures = new ju.HashMap[String, ju.Set[String]]
    val floatFeatures = new ju.HashMap[String, ju.Map[String, java.lang.Double]]
    val list: ju.Set[String] = new ju.HashSet[String]
    list.add("aaa")
    list.add("bbb")
    stringFeatures.put("strFeature1", list)
    val map = new ju.HashMap[String, java.lang.Double]
    map.put("lat", 37.7)
    map.put("long", 40.0)
    floatFeatures.put("loc", map)
    val featureVector: FeatureVector = new FeatureVector
    featureVector.setStringFeatures(stringFeatures)
    featureVector.setFloatFeatures(floatFeatures)
    featureVector
  }

  def makeConfig: String = {
    """
      | test_quantize {
      |   transform : quantize
      |   field1 : loc
      |   scale : 10
      |   output : loc_quantized
      | }
    """.stripMargin
  }

  @Test def testTransform {
    val config: Config = ConfigFactory.parseString(makeConfig)
    val pipeline = Transforms.pipeline(config, "test_quantize").get
    val featureVector: FeatureVector = makeFeatureVector
    val row = pipeline(featureVector.toRow)
    val newVector = row.toVector
    val stringFeatures = newVector.getStringFeatures
    assertTrue(stringFeatures.size == 2)
    val out: ju.Set[String] = stringFeatures.get("loc_quantized")
    assertTrue(out.size == 2)
    assertTrue(out.contains("lat=377"))
    assertTrue(out.contains("long=400"))
  }

  @Test
  def testScalaTransform(): Unit = {

    val pipeline: Transformer[Row, Row] = {

      val selector = VectorSelector("loc")
        .map(Scaler(3.0)
               .andThen(Scaler(1.5))
               .andThen(Scaler(4.2)))
      RowMapper(selector, VectorProducer("out"))
    }



    val vector = makeFeatureVector
    val newVector:FeatureVector = pipeline(vector.toRow).toVector
    assertEquals(newVector.getFloatFeatures.get("out").get("lat"), 37.7*3.0*1.5*4.2, 0.01)
    assertEquals(newVector.getFloatFeatures.get("out").get("long"), 40.0*3.0*1.5*4.2, 0.01)

    // Old strings are still there.
    assertTrue(newVector.stringFeatures.size == 1)
    assertEquals(newVector.stringFeatures.get("strFeature1"), ImmutableSet.of("aaa", "bbb"))
  }
}
