package com.airbnb.aerosolve.pipeline

import java.{lang, util}

import com.airbnb.aerosolve.core.FeatureVector
import org.apache.spark.ml.attribute._
import org.apache.spark.mllib.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.types._
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._

/**
 *
 */
object AerosolveSupport {

  implicit class FeatureVectorToRow(val vector: FeatureVector) {
    def toRow: Row = {
      if (vector == null) return null

      val floatPairs: Seq[(AttributeGroup, Vector)] = if (vector.getFloatFeatures != null) {
        vector.getFloatFeatures
          .asScala
          .map{ case (familyName, features) => {
            floatFamilyToVector(familyName, features)
          }}.toSeq
      } else Seq()

      val stringPairs: Seq[(AttributeGroup, Vector)] = if (vector.getStringFeatures != null) {
        vector.getStringFeatures
          .asScala
          .map{ case (familyName, features) => {
            stringFamilyToVector(familyName, features)
          }}.toSeq
      } else Seq()

      val densePairs:Seq[(AttributeGroup, Vector)] = if (vector.getDenseFeatures != null) {
        vector.getDenseFeatures
          .asScala
          .map { case (familyName, features) => {
            denseFamilyToVector(familyName, features)
          }}.toSeq
      } else Seq()

      val allPairs: Seq[(AttributeGroup, Vector)] = floatPairs ++ stringPairs ++ densePairs
      val values: Array[Any] = allPairs.map(_._2).toArray
      val fields = allPairs.map(_._1.toStructField())
      new AerosolveRow(values, StructType(fields))
    }
  }

  def denseFamilyToVector(familyName: String,
                          features: util.List[lang.Double]): (AttributeGroup, Vector) = {
    val innerVector = Vectors.dense(features.asScala.map(_.asInstanceOf[Double]).toArray)
    val group = new AttributeGroup(familyName, innerVector.size)
    (group, innerVector)
  }

  def floatFamilyToVector(familyName: String,
                          features: util.Map[String, lang.Double]): (AttributeGroup, Vector) = {
    val arr: Seq[(Attribute, Double)] = features
      .asScala
      .toSeq
      .map { case (key: String, value: lang.Double) => {
        floatPair(key, value)
      }
    }
    makeVectorPair(familyName, arr)
  }

  def floatPair(key: String, value: lang.Double): (NumericAttribute, Double) = {
    val field = NumericAttribute
      .defaultAttr
      .withName(key)
    (field, value.asInstanceOf[Double])
  }

  def stringFamilyToVector(familyName: String,
                           features: util.Set[String]): (AttributeGroup, Vector) = {
    val arr: Seq[(Attribute, Double)] = features
      .asScala
      .toSeq
      .map(key => {
        stringPair(key)
      })
    makeVectorPair(familyName, arr)
  }

  def stringPair(key: String): (NominalAttribute, Double) = {
    val attribute = NominalAttribute.defaultAttr
      .withName(key)
      .withValues(key)
    (attribute, 0.0)
  }

  private def makeVectorPair(familyName: String,
                             arr: Seq[(Attribute, Double)]): (AttributeGroup, Vector) = {
    val group = new AttributeGroup(familyName, arr.map(_._1).toArray)
    (group, Vectors.dense(arr.map(_._2).toArray))
  }

  implicit class RowToFeatureVector(val row: Row) {
    val DEFAULT_STRING_FAMILY = "DEFAULT_STRING"
    val DEFAULT_FLOAT_FAMILY = "DEFAULT_FLOAT"

    lazy val log = LoggerFactory.getLogger(getClass)

    def toVector: FeatureVector = {
      if (row == null) return null

      val vector: FeatureVector = new FeatureVector
      val floats = vector
        .setFloatFeatures(new java.util.HashMap[String, java.util.Map[String, java.lang.Double]]())
        .getFloatFeatures

      val strings = vector.setStringFeatures(new java.util.HashMap[String, java.util.Set[String]]())
        .getStringFeatures

      val denses = vector
        .setDenseFeatures(new java.util.HashMap[String, java.util.List[java.lang.Double]]())
        .getDenseFeatures

      def getOrCreateStringFamily(family: String) =
        strings.computeIfAbsent(family, toJavaFunction(
          (s: String) => new java.util.HashSet[String]()))

      def getOrCreateFloatFamily(family: String) =
        floats.computeIfAbsent(family, toJavaFunction(
          (s: String) => new java.util.HashMap[String, java.lang.Double]()))

      for (i <- 0 until row.length) {
        val field = row.schema(i)
        field.dataType match {
          case _: StringType => getOrCreateStringFamily(DEFAULT_STRING_FAMILY)
            .add(s"${field.name}:${row.getString(i)}")
          case _: NumericType => {
            Attribute.fromStructField(field) match {
              case numeric: NumericAttribute => getOrCreateFloatFamily(DEFAULT_FLOAT_FAMILY)
                .put(field.name, row.getDouble(i))
              case nominal: NominalAttribute => {
                val family = getOrCreateStringFamily(DEFAULT_STRING_FAMILY)
                val name = nominal
                  .getValue(row.getDouble(i).toInt)
                family.add(name)
              }
              case binary: BinaryAttribute => getOrCreateStringFamily(DEFAULT_STRING_FAMILY)
                .add(s"${field.name}:${if (row.getBoolean(i)) 'T' else 'F'}")
            }
          }
          case _: VectorUDT => {
            val group = AttributeGroup.fromStructField(field)
            val innerVector = row.get(i).asInstanceOf[Vector]
            val attributes = group.attributes.getOrElse(new Array[Attribute](0))
            val familyName = group.name
            if (attributes.length == 0 || (attributes(0).isNumeric && attributes(0).name.isEmpty)) {
              // dense
              val list: util.List[java.lang.Double] = innerVector.toArray
                .map(_.asInstanceOf[java.lang.Double]).toList.asJava
              denses.put(familyName, list)
            } else if (attributes(0).isNumeric) {
              // TODO (Brad): Can we mix and match nominal and numeric here?
              val family = getOrCreateFloatFamily(familyName)
              for (i <- attributes.indices) {
                family.put(attributes(i).name.get, innerVector(i).asInstanceOf[java.lang.Double])
              }
            } else if (attributes(0).isNominal) {
              val family = getOrCreateStringFamily(familyName)
              attributes.foreach(attr => family.add(
                attr.asInstanceOf[NominalAttribute]
                  .getValue(innerVector(i).toInt)))
            }
          }
          // I made the decision to ignore unknown types here because it made sense that we might
          // use more complex objects or UDTs up the pipeline. We'd just want to exclude them from
          // the final result. Having to remove them seems tedious and error prone. So I'm just
          // ignoring them here.  I'm commenting because I'm worried this will cause problems.
          case _ => log.warn("Unable to construct a FeatureVector with a field" +
                             s"of type ${field.dataType} named ${field.name}")
        }
      }
      vector
    }
  }

  def toJavaFunction[U, V](f:(U) => V): java.util.function.Function[U, V] =
    new java.util.function.Function[U, V] {
      override def apply(t: U): V = f(t)
    }

  def fromJavaFunction[U, V](f:java.util.function.Function[U,V]): (U) => V = f.apply

  // In case we need to add functionality later.
  type AerosolveRow = GenericRowWithSchema
}
