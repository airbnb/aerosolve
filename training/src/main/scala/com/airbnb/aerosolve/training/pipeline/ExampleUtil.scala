package com.airbnb.aerosolve.training.pipeline

import com.airbnb.aerosolve.core.features.Features
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

import scala.collection.mutable.ArrayBuffer

object ExampleUtil {
  def getFeatures(row: Row, schema: Array[StructField]) = {
    val features = Features.builder()
    val names = ArrayBuffer[String]()
    val values = ArrayBuffer[AnyRef]()

    for (i <- 0 until schema.length) {
      val rowSchema = schema(i)
      names += rowSchema.name
      if (!row.isNullAt(i)) {
        rowSchema.dataType match {
          case StringType => {
            values += row.getString(i)
          }
          case LongType => {
            values +=  row.getLong(i).toDouble.asInstanceOf[AnyRef]
          }
          case IntegerType => {
            values +=  row.getInt(i).toDouble.asInstanceOf[AnyRef]
          }
          case FloatType => {
            values += row.getFloat(i).toDouble.asInstanceOf[AnyRef]
          }
          case DoubleType => {
            values += row.getDouble(i).asInstanceOf[AnyRef]
          }
          case BooleanType => {
            values += row.getBoolean(i).asInstanceOf[AnyRef]
          }
          case _ => {
            // unknown type?
            assert(false)
          }
        }
      } else {
        values += null
      }
    }

    features.names(names.toArray)
    features.values(values.toArray)
    features.build()
  }
}
