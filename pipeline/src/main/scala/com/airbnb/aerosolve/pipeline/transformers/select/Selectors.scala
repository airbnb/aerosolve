package com.airbnb.aerosolve.pipeline.transformers.select

import com.airbnb.aerosolve.pipeline.AerosolveSchema
import com.airbnb.aerosolve.pipeline.transformers.{RowTransformer, Transformer}
import com.typesafe.config.Config
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructField

import scala.reflect.ClassTag

abstract class Selector[T: ClassTag] extends RowTransformer {
}