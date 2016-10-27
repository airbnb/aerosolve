package com.airbnb.aerosolve.training.photon.ml.data

import com.airbnb.aerosolve.core.Example
import org.apache.avro.generic.{GenericData, GenericRecord}
import org.apache.avro.{Schema, SchemaBuilder}

import scala.collection.immutable

import scala.collection.JavaConverters._

/**
  * A collection of utils functions used by PhotonML related work.
  *
  */
object PhotonMLUtils {
  // Example label family field name
  val LABEL = "label"

  // Avro namespace
  val AVRO_NAMESPACE = "com.airbnb.aerosolve.training.photon.ml.data.generated"

  // Avro schema fields
  val AVRO_UID = "uid"
  val AVRO_LABEL = "response"

  val AVRO_FEATURES = "features"
  val AVRO_NAME = "name"
  val AVRO_TERM = "term"
  val AVRO_VALUE = "value"
  val AVRO_META_DATA_MAP = "metadataMap"


  /**
    * Create a TrainingExampleAvro schema used by Photon ML:
    * [[https://github.com/linkedin/photon-ml#input-data-format]]
    *
    * @param name              optionally provide a record name for the schema. This is useful when multiple
    *                          Avro schemas are created and they needed to be distinguished.
    * @param featureNamespaces if an array of namespaces are provided, we will create a list of
    *                          extra feature arrays accordingly.
    * @return an Avro schema instance
    */
  def makeTrainingExampleAvroSchema(name: String = "TrainingExampleAvro",
                                    featureNamespaces: Array[String] = Array[String]()): Schema = {
    // Define feature item schema
    val featureItemSchema = SchemaBuilder.record("FeatureItem")
      .namespace(AVRO_NAMESPACE)
      .fields()
      .name(AVRO_NAME).`type`().stringType().noDefault()
      .name(AVRO_TERM).`type`().stringType().stringDefault("")
      .name(AVRO_VALUE).`type`().doubleType().noDefault()
      .endRecord()

    // Define overall training record schema
    val builder = SchemaBuilder.record(name)
      .namespace(AVRO_NAMESPACE)
      .fields()
      .name(AVRO_UID).`type`().nullable().stringType().noDefault()
      .name(AVRO_LABEL).`type`().doubleType().noDefault()
      .name(AVRO_META_DATA_MAP).`type`().map().values().longType().noDefault()

    // Global feature space, important to trigger noDefault() in the end otherwise, the field
    // will not be successfully committed into the builder
    builder.name(AVRO_FEATURES).`type`().array().items(featureItemSchema).noDefault()

    // Optionally, if feature family is provided, create per feature family feature space
    featureNamespaces.foreach { case namespace =>
      builder.name(makeAvroFeaturesFieldName(namespace))
        .`type`().array().items(featureItemSchema).noDefault()
    }

    builder.endRecord()
  }

  /**
    * Create an Avro feature record
    *
    * @param schema feature item schema, necessary to provide since schemas with different names or
    *               namespaces are considered to be different.
    * @param name   feature name
    * @param value  feature value
    * @param term   feature term (optionally to be empty)
    * @return the Avro feature item
    */
  def makeAvroFeatureItemRecord(schema: Schema,
                                name: String,
                                value: Double,
                                term: String = ""): GenericRecord = {
    val r = new GenericData.Record(schema)
    r.put(AVRO_NAME, name)
    r.put(AVRO_TERM, term)
    r.put(AVRO_VALUE, value)
    r
  }

  /**
    * Returns the avro feature array field name for a certain feature family
    *
    * @param familyName
    */
  def makeAvroFeaturesFieldName(familyName: String): String = familyName.toLowerCase + AVRO_FEATURES.capitalize

  /**
    * Create an Avro feature array
    *
    * @param example       Thrift example
    * @param schema   The avro record schema
    * @return The converted Avro feature array
    */
  def createAvroFeatureArrays(record: GenericData.Record,
                              example: Example,
                              schema: Schema,
                              familyBlacklist: immutable.Set[String] =
                              immutable.Set[String]("default_string", "bias", "miss", LABEL)
                             ): Unit = {
    val arraySchema = schema.getField(AVRO_FEATURES).schema()
    val featureSchema = arraySchema.getElementType()

    val featureNamespaces = schema.getFields().asScala
      .map(_.name())
      .filter(_.endsWith(AVRO_FEATURES.capitalize)) ++ Array[String](AVRO_FEATURES)

    for (fieldName <- featureNamespaces) {
      record.put(fieldName, new java.util.ArrayList[GenericRecord]())
    }

    val s = example.getExample().get(0)

    val denseFeatures = s.getDenseFeatures()
    if (denseFeatures != null) {
      val it = denseFeatures.entrySet().iterator()
      while (it.hasNext()) {
        val entry = it.next()

        val ns = entry.getKey()
        if (!familyBlacklist.contains(ns.toLowerCase())) {
          var i = 0
          val it2 = entry.getValue().iterator()

          val features = record.get(makeAvroFeaturesFieldName(ns))
            .asInstanceOf[java.util.List[GenericRecord]]

          while (it2.hasNext()) {
            val value = it2.next()
            features.add(makeAvroFeatureItemRecord(featureSchema, String.valueOf(i), value, ""))
            i += 1
          }

        }
      }
    }

    val floatFeatures = s.getFloatFeatures()
    if (floatFeatures != null) {
      val it = floatFeatures.entrySet().iterator()
      while (it.hasNext()) {
        val entry = it.next()

        val ns = entry.getKey()
        if (!familyBlacklist.contains(ns.toLowerCase())) {
          val features = record.get(makeAvroFeaturesFieldName(ns))
            .asInstanceOf[java.util.List[GenericRecord]]

          val it2 = entry.getValue().entrySet().iterator()
          while (it2.hasNext()) {
            val termValue = it2.next()

            val term = termValue.getKey()
            val value = termValue.getValue()
            features.add(makeAvroFeatureItemRecord(featureSchema, term, value, ""))
          }
        }
      }
    }

    val stringFeatures = s.getStringFeatures()
    if (stringFeatures != null) {
      val it = stringFeatures.entrySet().iterator()
      while (it.hasNext()) {
        val entry = it.next()

        val ns = entry.getKey()

        if (!familyBlacklist.contains(ns.toLowerCase())) {

          val features = record.get(makeAvroFeaturesFieldName(ns))
            .asInstanceOf[java.util.List[GenericRecord]]

          val it2 = entry.getValue().iterator()
          while (it2.hasNext()) {
            val term = it2.next()
            features.add(makeAvroFeatureItemRecord(featureSchema, term, 1.0d, ""))
          }
        }
      }
    }
  }
}
