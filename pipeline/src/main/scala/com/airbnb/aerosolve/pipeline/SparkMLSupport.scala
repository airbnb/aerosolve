package com.airbnb.aerosolve.pipeline

import com.airbnb.aerosolve.pipeline.AerosolveSupport.AerosolveRow
import com.airbnb.aerosolve.pipeline.transformers.Transformer
import org.apache.spark.ml.attribute.{AttributeGroup, NominalAttribute, Attribute}
import org.apache.spark.mllib.linalg.{Vectors, Vector, VectorUDT}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext, Row}
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types.{MetadataBuilder, NumericType, StructType, StructField}

/**
 *
 */
object SparkMLSupport {

  implicit class DataFramer(rdd: RDD[Row])  {
    def toDF(sc: SQLContext): DataFrame = {
      sc.createDataFrame(rdd, rdd.first().schema)
    }

    def collapseSchemaToDF(sc: SQLContext): DataFrame = {
      val schema = rdd.map(_.schema).reduce((type1, type2) =>
                                              SparkMLSupport.combineSchemas(type1, type2))
      val newRdd = rdd.map(row => SparkMLSupport.reorganize(row, schema))
      sc.createDataFrame(newRdd, schema)
    }
  }

  implicit def dataFrameToRowRDD(dataframe: DataFrame): RDD[Row] = {
    dataframe.rdd
  }

  implicit class FieldTransformer[T](valueTransformer: Transformer[T, T])
    extends Transformer[(StructField, T), (StructField, T)] {

    override def apply(pair: (StructField, T)): (StructField, T) = {
      val (field, value) = pair
      (field, valueTransformer(value))
    }
  }

  def mergeFields(field: StructField, otherField: StructField): StructField = {
    otherField match {
      case null => field
      case secondField if secondField.dataType == field.dataType =>
        field.dataType match {
          case _: VectorUDT => {
            // If it's a vector, we need to merge the AttributeGroups
            mergeGroups(field, secondField)
          }
          case _: NumericType => {
            // If it's Numeric, we need to merge the Attribute values.
            val attribute = Attribute.fromStructField(field)
            val otherAttribute = Attribute.fromStructField(secondField)
            val newAttribute: NominalAttribute = mergeAttributes(attribute, otherAttribute)
            newAttribute.toStructField(field.metadata)
          }
          case _ => field
        }
      case secondField => throw new IllegalStateException(
        s"""Trying to merge two fields with the same name (${field.name} but different
           | types: ${field.dataType} : ${secondField.dataType}"""
          .stripMargin
          .replaceAll("\n", " "))
    }
  }

  def mergeGroups(field: StructField, otherField: StructField): StructField = {
    val group = AttributeGroup.fromStructField(field)
    val otherGroup = AttributeGroup.fromStructField(otherField)
    require(group.name == otherGroup.name, "When merging schemas, two fields had the same" +
                                           "name but the groups did not.")
    if (group.attributes.isEmpty) {
      require(group.numAttributes.get == otherGroup.numAttributes.get)
      return field
    }
    if (group.attributes.get.apply(0).name.isEmpty) {
      require(group.attributes.get.length == otherGroup.attributes.get.length)
      return field
    }

    val attributes = mergeAttributeArrays(group.attributes.get, otherGroup.attributes.get)
    new AttributeGroup(group.name, attributes).toStructField()
  }
  
  def mergeAttributes(attribute: Attribute, otherAttribute: Attribute): NominalAttribute = {
    require(attribute.attrType == otherAttribute.attrType,
            "Trying to merge different types of attributes: " +
            s"${attribute.name}:${attribute.attrType} :: " +
            s"${otherAttribute.name}:${otherAttribute.attrType}")
    // TODO (Brad): Merge min, max, etc. of numerics?
    val newAttribute = attribute match {
      case nominal: NominalAttribute => {
        val values =
          nominal
            .values
            .map(_.toSet)
            .getOrElse(Set()) ++
          otherAttribute
            .asInstanceOf[NominalAttribute]
            .values
            .map(_.toSet)
            .getOrElse(Set())

        nominal
          .withValues(values.toArray)
      }
    }
    newAttribute
  }

  def combineSchemas(structType: StructType, structType2: StructType): StructType = {
    val diff = structType2.toSet.diff(structType.toSet)

    val fields = diff.map(field => {
      mergeFields(structType(field.name), field)
    }).toSeq
    StructType(structType.fields ++ fields)
  }

  def mergeAttributeArrays(array: Array[Attribute], otherArray: Array[Attribute]):
  Array[Attribute] = {
    val attributesMap: Map[String, Attribute] = mapAttributesByName(array)
    val otherAttributesMap: Map[String, Attribute] = otherArray
      .flatMap(attr => attr
        .name
        .map(name => (name, attributesMap.get(name)
          .map(other => mergeAttributes(attr, other))
          .getOrElse(attr))))
      .toMap
    (attributesMap ++ otherAttributesMap).values.toArray
  }

  def mapAttributesByName(values: Array[Attribute]): Map[String, Attribute] = {
    values
      .flatMap(attr => attr
        .name
        .map(name => (name, attr)))
      .toMap
  }

  def reorganize(row: Row, schema: StructType): Row = {
    val arr = Array[Any](schema.length)
    val otherSet = schema.fieldNames.toSet
    row
      .schema
      .indices
      .foreach(i => {
        val field = row.schema(i)
        // fieldIndex throws an exception if name doesn't exist and we don't want that.
        if (otherSet.contains(field.name)) {
          val newFieldIndex = schema.fieldIndex(field.name)
          arr(newFieldIndex) = reorganizeValue(field, schema(newFieldIndex), row(i))
        }
      })
    new AerosolveRow(arr, schema)
  }

  def reorganizeValue(field: StructField, newField: StructField, value: Any): Any = {
    require(field.dataType == newField.dataType)

    field.dataType match {
      case _: VectorUDT =>
        reorganizeVector(
          AttributeGroup.fromStructField(field),
          AttributeGroup.fromStructField(newField),
          value.asInstanceOf[Vector])
      case _: NumericType => {
        val attribute = Attribute.fromStructField(newField)
        attribute match {
          case nominal: NominalAttribute if nominal.values.isDefined => {
            val existingNominal = Attribute.fromStructField(field).asInstanceOf[NominalAttribute]
            val name = existingNominal.values.get(value.asInstanceOf[Double].toInt)
            nominal.indexOf(name).toDouble
          }
          case _ => value
        }
      }
      case _ => value
    }
  }

  def reorganizeVector(group: AttributeGroup, newGroup: AttributeGroup, vector: Vector): Vector = {
    // We can't reorganize if there are no named attributes.
    if (group.attributes.isEmpty || newGroup.attributes.isEmpty) {
      // TODO (Brad): Maybe we should expand the size?
      return vector
    }

    reorganizeVector(group.attributes.get, newGroup.attributes.get, vector)
  }

  def reorganizeVector(attributes: Array[Attribute], newAttributes: Array[Attribute],
                       vector: Vector): Vector = {
    val nameToIndex = newAttributes
      .zipWithIndex
      .map { case (attr: Attribute, index: Int) => (attr.name.get, index) }
      .toMap

    val values = Array.fill[Double](nameToIndex.size)(0.0)

    attributes
      .zipWithIndex
      .foreach { case (attribute, index) => {
        nameToIndex
          .get(attribute.name.get)
          .foreach(newIndex => values(newIndex) = vector(index))
      }
               }

    Vectors.dense(values).compressed
  }
}
