package com.airbnb.aerosolve.training.photon.ml.data

import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.core.features.Features
import org.apache.avro.Schema
import org.apache.avro.generic.{GenericData, GenericRecord}
import org.junit.Assert._
import org.junit.Test

import scala.collection.JavaConverters._

/**
  * This class tests [[PhotonMLUtils]].
  *
  */
class PhotonMLUtilsTest {

  import PhotonMLUtils._

  private val EPS = 1e-7

  private def verifyBasicTrainingExampleAvroFields(schema: Schema): Unit = {
    assertEquals(schema.getNamespace(), AVRO_NAMESPACE)

    val fields = schema.getFields().asScala

    val namesSet = fields.map(_.name())
    assertTrue(namesSet.size >= 4)
    assertTrue(namesSet.contains(AVRO_UID))
    assertTrue(namesSet.contains(AVRO_LABEL))
    assertTrue(namesSet.contains(AVRO_META_DATA_MAP))
    assertTrue(namesSet.contains(AVRO_FEATURES))

    // Verify ordinary fields
    val nullableUid = schema.getField(AVRO_UID).schema()
    assertEquals(nullableUid.getType(), Schema.Type.UNION)
    assertEquals(nullableUid.getTypes().asScala.map(_.getType()).toSet,
      Set[Schema.Type](Schema.Type.NULL, Schema.Type.STRING))

    assertEquals(schema.getField(AVRO_LABEL).schema().getType(), Schema.Type.DOUBLE)
    val mapSchema = schema.getField(AVRO_META_DATA_MAP).schema()
    assertEquals(mapSchema.getType(), Schema.Type.MAP)
    assertEquals(mapSchema.getValueType().getType(), Schema.Type.LONG)

    // Verify the default feature array field
    val featureSchema = schema.getField(AVRO_FEATURES).schema().getElementType()
    verifyFeatureItemSchema(featureSchema)

    // Verify other fields (assuming all should be feature array type)
    fields.filter(f => !namesSet.contains(f.name())).foreach { case featureField =>
      assertTrue(featureField.name().endsWith(AVRO_FEATURES.capitalize))
      verifyFeatureItemSchema(featureField.schema().getElementType())
    }
  }

  private def verifyFeatureItemSchema(featureSchema: Schema): Unit = {
    val fields = featureSchema.getFields().asScala

    val namesSet = fields.map(_.name())
    assertEquals(namesSet.size, 3)
    assertTrue(namesSet.contains(AVRO_NAME))
    assertTrue(namesSet.contains(AVRO_TERM))
    assertTrue(namesSet.contains(AVRO_VALUE))

    assertEquals(featureSchema.getField(AVRO_NAME).schema().getType(), Schema.Type.STRING)
    assertEquals(featureSchema.getField(AVRO_TERM).schema().getType(), Schema.Type.STRING)
    assertEquals(featureSchema.getField(AVRO_VALUE).schema().getType(), Schema.Type.DOUBLE)
  }

  @Test
  def testMakeTrainingExampleAvroSchema(): Unit = {
    val schema1 = makeTrainingExampleAvroSchema()
    assertEquals(schema1.getName(), "TrainingExampleAvro")
    verifyBasicTrainingExampleAvroFields(schema1)


    val schema2 = makeTrainingExampleAvroSchema(name = "TrainingExampleAvroV2")
    assertEquals(schema2.getName(), "TrainingExampleAvroV2")
    verifyBasicTrainingExampleAvroFields(schema2)


    val schema3 = makeTrainingExampleAvroSchema(name = "TrainingExampleAvroV3",
      featureNamespaces = Array[String]("a", "b", "c"))
    assertEquals(schema3.getName(), "TrainingExampleAvroV3")
    verifyBasicTrainingExampleAvroFields(schema3)
    assertEquals(schema3.getFields().size, 7)
    verifyFeatureItemSchema(schema3.getField("a" +
      AVRO_FEATURES.capitalize).schema().getElementType())
    verifyFeatureItemSchema(schema3.getField("b" +
      AVRO_FEATURES.capitalize).schema().getElementType())
    verifyFeatureItemSchema(schema3.getField("c" +
      AVRO_FEATURES.capitalize).schema().getElementType())
  }

  @Test
  def testMakeAvroFeatureItemRecord(): Unit = {
    val featureSchema = makeTrainingExampleAvroSchema().getField(AVRO_FEATURES)
      .schema().getElementType()

    val r = makeAvroFeatureItemRecord(featureSchema, "foo", 1.2d)
    assertEquals("foo", r.get(AVRO_NAME))
    assertTrue(r.get(AVRO_TERM).toString().isEmpty())
    assertEquals(1.2d, r.get(AVRO_VALUE).asInstanceOf[Double], EPS)
  }

  @Test
  def testMakeAvroFeaturesFieldName(): Unit = {
    assertEquals("fooFeatures", makeAvroFeaturesFieldName("foo"))
    assertEquals("Features", makeAvroFeaturesFieldName(""))
    assertEquals("bar_aFeatures", makeAvroFeaturesFieldName("bar_A"))
  }


  private def mockExample(): Example = {
    val builder = Features.builder()

    builder.names(Array[String]("foo_a",
      "foo_b",
      "bar_c",
      "tar_d"))

    builder.values(Array[Object](
      new java.lang.Double(1.0d),
      new java.lang.Long(-2L),
      new java.lang.Integer(5),
      "a"
    ))

    builder.build().toExample(false)
  }

  private def mockExampleWithMissingFields(): Example = {
    val builder = Features.builder()

    builder.names(Array[String]("foo_a",
      "bar_c"))

    builder.values(Array[Object](
      new java.lang.Double(1.0d),
      new java.lang.Integer(5)
    ))

    builder.build().toExample(false)
  }

  @Test
  def testCreateAvroFeatureArrays(): Unit = {
    val schema = makeTrainingExampleAvroSchema(
      featureNamespaces = Array[String]("foo", "bar", "tar"))

    val record = new GenericData.Record(schema)

    val example = mockExample()
    println(example.toString())

    createAvroFeatureArrays(record, example, schema)

    val features = record.get(AVRO_FEATURES).asInstanceOf[java.util.List[GenericRecord]]
    assertTrue(features.isEmpty())

    val fooFeatures = record.get(makeAvroFeaturesFieldName("foo"))
      .asInstanceOf[java.util.List[GenericRecord]]
    assertEquals(fooFeatures.size(), 2)
    val fooFL = fooFeatures.iterator().asScala.toSeq.sortBy(_.get(AVRO_NAME).toString())
    assertEquals(fooFL(0).get(AVRO_NAME), "a")
    assertEquals(fooFL(0).get(AVRO_TERM), "")
    assertEquals(fooFL(0).get(AVRO_VALUE).asInstanceOf[Double], 1.0, EPS)
    assertEquals(fooFL(1).get(AVRO_NAME), "b")
    assertEquals(fooFL(1).get(AVRO_TERM), "")
    assertEquals(fooFL(1).get(AVRO_VALUE).asInstanceOf[Double], -2.0, EPS)



    val barFeatures = record.get(makeAvroFeaturesFieldName("bar"))
      .asInstanceOf[java.util.List[GenericRecord]]
    assertEquals(barFeatures.size(), 1)
    assertEquals(barFeatures.get(0).get(AVRO_NAME), "c")
    assertEquals(barFeatures.get(0).get(AVRO_TERM), "")
    assertEquals(barFeatures.get(0).get(AVRO_VALUE).asInstanceOf[Double], 5d, EPS)

    val tarFeatures = record.get(makeAvroFeaturesFieldName("tar"))
      .asInstanceOf[java.util.List[GenericRecord]]
    assertEquals(tarFeatures.size(), 1)
    assertEquals(tarFeatures.get(0).get(AVRO_NAME), "d:a")
    assertEquals(tarFeatures.get(0).get(AVRO_TERM), "")
    assertEquals(tarFeatures.get(0).get(AVRO_VALUE).asInstanceOf[Double], 1d, EPS)


    // Test if some fields are not presented in the schema
    val exampleMissing = mockExampleWithMissingFields()
    val record2 = new GenericData.Record(schema)
    createAvroFeatureArrays(record2, exampleMissing, schema)
    val fooFeatures2 = record2.get(makeAvroFeaturesFieldName("foo"))
      .asInstanceOf[java.util.List[GenericRecord]]
    assertEquals(fooFeatures2.size(), 1)
    assertEquals(fooFeatures2.get(0).get(AVRO_NAME), "a")
    assertEquals(fooFeatures2.get(0).get(AVRO_TERM), "")
    assertEquals(fooFeatures2.get(0).get(AVRO_VALUE).asInstanceOf[Double], 1d, EPS)

    val barFeatures2 = record.get(makeAvroFeaturesFieldName("bar"))
      .asInstanceOf[java.util.List[GenericRecord]]
    assertEquals(barFeatures2.size(), 1)
    assertEquals(barFeatures2.get(0).get(AVRO_NAME), "c")
    assertEquals(barFeatures2.get(0).get(AVRO_TERM), "")
    assertEquals(barFeatures2.get(0).get(AVRO_VALUE).asInstanceOf[Double], 5d, EPS)
  }
}
