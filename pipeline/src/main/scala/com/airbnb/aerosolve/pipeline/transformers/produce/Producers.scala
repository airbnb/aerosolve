package com.airbnb.aerosolve.pipeline.transformers.produce

import com.airbnb.aerosolve.pipeline.AerosolveSupport.AerosolveRow
import com.airbnb.aerosolve.pipeline.{SparkMLSupport, AerosolveSchema, AerosolveSupport}
import com.typesafe.config.Config
import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructType, MetadataBuilder, StructField}
import workflow.Transformer

import scala.reflect.ClassTag

abstract class Producer[T:ClassTag] extends Transformer[(Row, Row), Row]