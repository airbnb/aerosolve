package com.airbnb.aerosolve.pipeline.transformers.nominal

import com.airbnb.aerosolve.pipeline.AerosolveSupport
import com.airbnb.aerosolve.pipeline.transformers.Transformer
import org.apache.spark.ml.attribute._
import org.apache.spark.sql.types._

/**
 *
 */
case class Renamer(template: String, includeValue: Boolean = true)
  extends Transformer[(StructField, Double), (StructField, Double)] {

  override def apply(in: (StructField, Double)): (StructField, Double) = {
    val (field, value) = in
    if (includeValue) {
      val newName = template.format(field.name, value)
      (field.copy(name = newName), value)
    } else {
      val fieldName = AerosolveSupport.stripFamilyName(field.name, field.metadata)
      val newName = template.format(fieldName, value)
      val nominal = NominalAttribute.defaultAttr
        .withName(newName)
        .withValues(newName)
      (nominal.toStructField(field.metadata), 0.0)
    }
  }
}
