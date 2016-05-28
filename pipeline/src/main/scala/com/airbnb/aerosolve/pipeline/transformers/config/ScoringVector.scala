package com.airbnb.aerosolve.pipeline.transformers.config

import breeze.linalg.DenseVector
import org.apache.spark.ml.attribute._
import org.apache.spark.mllib.linalg.{Vector, VectorUDT}
import org.apache.spark.sql.catalyst.expressions.{BaseGenericInternalRow, MutableRow}
import org.apache.spark.sql.types._

import scala.collection.mutable.ArrayBuffer

/**
 *
 */

object ScoringVector {
  def generateSchema(schema: StructType): ScoringSchema = {
    val buf = ArrayBuffer[Attribute]()
    val arr = Array[Int](schema.length)
    var schemaIndex = 0
    schema.foreach(field => {
      arr(schemaIndex) = buf.length
      if (field.dataType == new VectorUDT) {
        val group = AttributeGroup.fromStructField(field)
        val groupAttributes = group.attributes.get
        buf ++= groupAttributes
      } else {
        buf += Attribute.fromStructField(field)
      }
      schemaIndex += 1
    })
    arr(schemaIndex) = buf.length
    ScoringSchema(schema, arr, buf.toArray)
  }
}

class ScoringVector(val schema: ScoringSchema, val values: Array[Double]) extends MutableRow with BaseGenericInternalRow {

  def this(schema: ScoringSchema) = this(schema, Array.fill[Double](schema.attributes.length)(0.0))
  def this(rowSchema: StructType) = this(ScoringVector.generateSchema(rowSchema))

  val innerVector: DenseVector[Double] = DenseVector(values)

  override def setNullAt(i: Int): Unit = {
    val index = schema.vectorIndices(i)
    innerVector(index) = 0.0
  }

  // TODO (Brad): This is fast but it probably needs a lot more validation.
  override def update(i: Int, value: Any): Unit = {
    val index = schema.vectorIndices(i)
    value match {
      case value: Double => innerVector(index) = value
      case str: String => innerVector(index) = 1.0
      case vector: breeze.linalg.Vector[Double] => for (i <- 0 until vector.size) {
        innerVector(index + i) = vector(i)
      }
      case mlVector: Vector =>  for (i <- 0 until mlVector.size) {
        innerVector(index + i) = mlVector(i)
      }
      case value: Number => innerVector(index) = value.doubleValue()
      case value: Int => innerVector(index) = value.toDouble
      case value: Long => innerVector(index) = value.toDouble
      case bool: Boolean => innerVector(index) = if (bool) 1.0 else 0.0
      case _ => throw new IllegalArgumentException(s"Can't add a value of type ${value.getClass}" +
        " to a ScoringVector")
    }

  }

  override protected def genericGet(ordinal: Int): Any = {
    val index = schema.vectorIndices(ordinal)
    innerVector(index)
  }

  override def copy(): ScoringVector = new ScoringVector(schema, innerVector.toArray)

  override def numFields: Int = schema.rowSchema.length
}

case class ScoringSchema(rowSchema: StructType, vectorIndices: Array[Int],
                         attributes: Array[Attribute])

