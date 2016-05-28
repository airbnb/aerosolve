package com.airbnb.aerosolve.pipeline.transformers

import com.airbnb.aerosolve.pipeline.AerosolveSupport.AerosolveRow
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row}

import scala.reflect.ClassTag

/**
 *
 */
object RowTransformer {
  implicit def transformerToRow(transformer: FieldTransformer):
    RowTransformer = {
    RowTransformer(valueTransformer = Some(transformer))
  }

  implicit def transformerToRow(transformer: Transformer[_, _]):
    RowTransformer = {
    RowTransformer(valueTransformer = Some(FieldValueTransformer(transformer)))
  }
}

abstract class SchemaTransformer extends Transformer[Row, Row] with SchemaAdapter {}

trait SchemaAdapter {
  def transformSchema(schema: StructType): StructType
}

object FieldTransformer {
  def pipeline(left: FieldTransformer, right: FieldTransformer): FieldTransformer = {
    FieldTransformerPipeline(left, right)
  }
}

case class FieldTransformerPipeline(left: FieldTransformer, right: FieldTransformer)
  extends FieldTransformer {

  override def transformSchema(schema: StructType): StructType =
    right.transformSchema(left.transformSchema(schema))

  override def apply(input: (StructField, Any)) = right(left(input))
}

abstract class FieldTransformer extends Transformer[(StructField, Any), (StructField, Any)]
  with SchemaAdapter {

  def andThen(t: FieldTransformer): FieldTransformer = FieldTransformerPipeline(this, t)
}

abstract class Producer extends Transformer[(Row, Row), Row] with SchemaAdapter {}

abstract class Selector extends SchemaTransformer {}

case class RowTransformer(
    selector: Option[Selector] = None,
    valueTransformer: Option[FieldTransformer] = None,
    producer: Option[Producer] = None)
  extends SchemaTransformer {

  override def transformSchema(schema: StructType): StructType = {
    val inputSchema = selector.fold(schema)(_.transformSchema(schema))
    val transformedSchema = valueTransformer.fold(inputSchema)(_.transformSchema(inputSchema))
    producer.fold(transformedSchema)(_.transformSchema(transformedSchema))
  }

  def apply(dataset: DataFrame): RDD[Row] = {
    apply(dataset.rdd)
  }

  override def apply(row: Row) : Row = {
    val inputRow = selector.fold(row)(_.apply(row))
    val transformedRow = valueTransformer.fold(inputRow)(t => {
      val transformed = inputRow
        .toSeq
        .zip(inputRow.schema)
        .map{ case (value, field) => t.apply((field, value)) }
      new AerosolveRow(transformed.map(_._2).toArray, StructType(transformed.map(_._1).toArray))
    })
    producer.fold(transformedRow)(_.apply((transformedRow, row)))
  }

  def map(newTransformer: FieldTransformer): RowTransformer = {
    // TODO (Brad): Calculate new schema
    RowTransformer(selector,
                  Some(valueTransformer.fold(newTransformer)(t =>
                         FieldTransformer.pipeline(t, newTransformer))),
                  producer)
  }

  def mapValues(transformer: Transformer[_, _]): RowTransformer = {
    val transformer: FieldTransformer = FieldValueTransformer(transformer)
    RowTransformer(selector,
                   Some(valueTransformer.fold(transformer)(t =>
                         FieldTransformer.pipeline(t, transformer))),
                   producer)
  }
}

case class FieldValueTransformer[I: ClassTag](transformer: Transformer[I, _])
  extends FieldTransformer {

  override def transformSchema(schema: StructType): StructType = schema

  def apply(input: (StructField, Any)): (StructField, Any) = {
    val newValue = transformer(input._2.asInstanceOf[I])
    (input._1, newValue)
  }
}