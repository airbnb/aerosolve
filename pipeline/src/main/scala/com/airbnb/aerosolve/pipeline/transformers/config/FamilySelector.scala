package com.airbnb.aerosolve.pipeline.transformers.config

import java.util

import com.airbnb.aerosolve.core.FeatureVector
import com.airbnb.aerosolve.pipeline.AerosolveSupport
import com.airbnb.aerosolve.pipeline.AerosolveSupport.AerosolveRow
import com.airbnb.aerosolve.pipeline.transformers.Transformer
import com.typesafe.config.Config
import org.apache.spark.ml.attribute.NumericAttribute
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructField, StructType}

import scala.collection.JavaConverters._

/**
 *
 */
object FamilySelector {
  def apply(config: Config,
            familyType: String,
            familyField:String = "field1",
            featuresField: String = "keys",
            featureField: String = "value1",
            otherFeatureField: String = "value2"): FamilySelector[_] = {
    val family = config.getString(familyField)

    val features: Option[Set[String]] = if (config.hasPath(featuresField)) {
      Some(config.getStringList(featuresField).asScala.toSet)
    } else if (config.hasPath(featureField)) {
      val baseSet = if (config.hasPath(otherFeatureField)) {
        Set(config.getString(otherFeatureField))
      } else Set[String]()
      Some(baseSet + config.getString(featureField))
    } else None

    val excludedFeatures = if (config.hasPath("exclude_features")) {
      Some(config.getStringList("exclude_features").asScala.toSet)
    } else None

    familyType match {
      case "string" => StringFamilySelector(family, features, excludedFeatures)
      case "float" => FloatFamilySelector(family, features, excludedFeatures)
      case "dense" => {
        require(features.isEmpty, "Cannot select features from a dense vector")
        require(excludedFeatures.isEmpty, "Cannot exclude features from a dense vector")
        DenseFamilySelector(family)
      }
    }

  }
}

abstract class FamilySelector[T](family: String,
                              features: Option[Set[String]],
                              excludedFeatures: Option[Set[String]])
  extends Transformer[FeatureVector, Row] {

  val includedFeatures: Option[Set[String]] =
    features.map(fs => fs -- excludedFeatures.getOrElse(Set()))

  override def apply(featureVector: FeatureVector) : Row = {
    val opt: Option[Iterable[(StructField, Double)]] = Option(getMap(featureVector))
      .flatMap(fam => Option(fam.get(family)))
      .map(collection => includedFeatures
        .map(_.flatMap(key => extractFeature(collection, key)))
        .getOrElse(filterFeatures(collection, excludedFeatures.getOrElse(Set[String]()))))

    opt.map(features => {
      val values = features.map(_._2).toArray[Any]
      val schema = StructType(features.map(_._1).toArray)
      new AerosolveRow(values, schema)
    }).orNull
  }

  def getMap(vector: FeatureVector): util.Map[String, T]

  def extractFeature(collection: T, key: String): Option[(StructField, Double)]

  def filterFeatures(collection: T, excluded: Set[String]): Iterable[(StructField, Double)]
}

case class StringFamilySelector(family: String,
                                features: Option[Set[String]],
                                excludedFeatures: Option[Set[String]])
  extends FamilySelector[util.Set[String]](family, features, excludedFeatures) {

  override def getMap(vector: FeatureVector): util.Map[String, util.Set[String]] =
    vector.getStringFeatures

  override def filterFeatures(collection: util.Set[String],
                              excluded: Set[String]): Iterable[(StructField, Double)] = {
    (collection.asScala -- excluded).map(s => makePair(s))
  }

  override def extractFeature(collection: util.Set[String],
                              key: String): Option[(StructField, Double)] = {
    if (collection.contains(key)) {
      Some(makePair(key))
    } else None
  }

  def makePair(s: String): (StructField, Double) = {
    val pair = AerosolveSupport.stringPair(s)
    (pair._1.toStructField(), pair._2)
  }

}

case class FloatFamilySelector(family: String,
                                features: Option[Set[String]],
                                excludedFeatures: Option[Set[String]])
  extends FamilySelector[util.Map[String, java.lang.Double]](family, features, excludedFeatures) {

  override def getMap(vector: FeatureVector): util.Map[String, util.Map[String, java.lang.Double]] =
    vector.getFloatFeatures

  override def filterFeatures(collection: util.Map[String, java.lang.Double],
                              excluded: Set[String]): Iterable[(StructField, Double)] = {
    (collection.asScala -- excluded).map(pair => makePair(pair._1, pair._2))
  }

  override def extractFeature(collection: util.Map[String, java.lang.Double],
                              key: String): Option[(StructField, Double)] = {
    if (collection.containsKey(key)) {
      Some(makePair(key, collection.get(key)))
    } else None
  }

  def makePair(s: String, value: Double): (StructField, Double) = {
    val pair = AerosolveSupport.floatPair(s, value)
    (pair._1.toStructField(), pair._2)
  }

}

case class DenseFamilySelector(family: String)
  extends FamilySelector[util.List[java.lang.Double]](family, None, None) {

  override def getMap(vector: FeatureVector): util.Map[String, util.List[java.lang.Double]] =
    vector.getDenseFeatures

  override def filterFeatures(collection: util.List[java.lang.Double],
                              excluded: Set[String]): Iterable[(StructField, Double)] = {

    collection.asScala.zipWithIndex.map(pair => makePair(pair))
  }

  // Not implemented on purpose.
  override def extractFeature(collection: util.List[java.lang.Double],
                              key: String): Option[(StructField, Double)] = ???

  def makePair(input: (java.lang.Double, Int)): (StructField, Double) = {
    val field = NumericAttribute.defaultAttr.withIndex(input._2).toStructField()
    (field, input._1.asInstanceOf[Double])
  }

}

