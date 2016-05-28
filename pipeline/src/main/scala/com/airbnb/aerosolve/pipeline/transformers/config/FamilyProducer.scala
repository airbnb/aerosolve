package com.airbnb.aerosolve.pipeline.transformers.config

import com.airbnb.aerosolve.core.FeatureVector
import com.airbnb.aerosolve.pipeline.transformers.Transformer
import com.airbnb.aerosolve.pipeline.transformers.config.FamilyProducer.FamilyProducer
import com.airbnb.aerosolve.pipeline.{AerosolveSupport, SparkMLSupport}
import com.typesafe.config.Config
import org.apache.spark.ml.attribute.{NominalAttribute, NumericAttribute, Attribute}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

import scala.reflect.ClassTag
import java.util
/**
 *
 */
object FamilyProducer {

  def apply(config: Config,
            familyType: String,
            familyField:String = "output"): FamilyProducer = {
    val family = config.getString(familyField)
    familyType match {
      case "string" => StringFamilyProducer(family)
      case "float" => FloatFamilyProducer(family)
      case "dense" => DenseFamilyProducer(family)
    }
  }

  type FamilyProducer = Transformer[(Row, FeatureVector), FeatureVector]
}



// TODO (Brad): Handle Vectors in all of these?
case class FloatFamilyProducer(family: String) extends FamilyProducer {

  override def apply(input: (Row, FeatureVector)): FeatureVector = {
    val (row, vector) = input
    val fields = row.schema.toArray
    if (vector.getFloatFeatures == null) {
      vector.setFloatFeatures(new util.HashMap[String, util.Map[String, java.lang.Double]])
    }
    val floatMap = vector.getFloatFeatures.computeIfAbsent(family, AerosolveSupport.toJavaFunction(
      (s: String) => new java.util.HashMap[String, java.lang.Double]()))
    for (i <- fields.indices) {
      // Assuming it's numeric.
      floatMap.put(fields(i).name, row.getDouble(i))
    }
    vector
  }
}

case class StringFamilyProducer(family: String) extends FamilyProducer {

  override def apply(input: (Row, FeatureVector)): FeatureVector = {
    val (row, vector) = input
    val fields = row.schema.toArray
    if (vector.getStringFeatures == null) {
      vector.setStringFeatures(new util.HashMap[String, util.Set[String]])
    }
    val stringSet = vector.getStringFeatures.computeIfAbsent(family, AerosolveSupport.toJavaFunction(
      (s: String) => new java.util.HashSet[String]()))
    for (i <- fields.indices) {
      val field = fields(i)
      field.dataType match {
        case _:NumericType => {
          Attribute.fromStructField(field) match {
            case nominal: NominalAttribute => if (nominal.values.isDefined) {
              nominal.values.get(row.getDouble(i).toInt)
            } else {
              stringSet.add(field.name)
            }
            case _ => stringSet.add(field.name)
          }
        }
        case _: StringType => stringSet.add(row.getString(i))
      }
    }
    vector
  }
}

case class DenseFamilyProducer(family: String) extends FamilyProducer {

  override def apply(input: (Row, FeatureVector)): FeatureVector = {
    val (row, vector) = input
    if (vector.getDenseFeatures == null) {
      vector.setDenseFeatures(new util.HashMap[String, util.List[java.lang.Double]])
    }
    val list = new util.ArrayList[java.lang.Double](row.schema.length)
    for (i <- 0 until row.schema.length) {
      list.add(row.getDouble(i))
    }
    vector.getDenseFeatures.put(family, list)
    vector
  }
}

  /*def apply(values: Array[T], schema: AerosolveSchema, row: Row) = {
    // TODO (Brad): handle MutableRow for efficiency
    new AerosolveRow((row.toSeq ++ values).toArray, schema)
  }

  def transformSchema(schema: AerosolveSchema, fields: Seq[Array[StructField]]):
  AerosolveSchema = {
    val newSchema = if (squashSchema) {
      fields
        .map(StructType.apply)
        .reduce(SparkMLSupport.combineSchemas)
    } else {
      StructType(fields.head)
    }

    schema.addFamily(family, newSchema.fields.map(changeFamily))
  }

  def changeFamily(field: StructField): StructField = {
    val metadata = new MetadataBuilder()
      .withMetadata(field.metadata)
      .putString(AerosolveSupport.FAMILY_NAME, family)
      .build()
    val name = AerosolveSupport.stripFamilyName(field.name, metadata)
    field.copy(name = name, metadata = metadata)
  }
}*/
