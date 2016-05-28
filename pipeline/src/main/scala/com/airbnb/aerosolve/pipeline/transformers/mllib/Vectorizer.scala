package com.airbnb.aerosolve.pipeline.transformers.mllib

import org.apache.spark.mllib.linalg.Vector
import nodes.util.AllSparseFeatures
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import workflow.{Estimator, Transformer}

/**
 *
 */
object Vectorizer {
  val LABEL_COLUMN = "label"
  val FEATURES_COLUMN = "features"
}

case class Vectorizer(labelFamily: Option[String] = None) extends Estimator[Row, Row] {

  override protected def fit(data: RDD[Row]): Transformer[Row, Row] = {
    val labels = labelFamily
      .map(family => data
        .map(row => row
          .get(row.
            fieldIndex(family))
          .asInstanceOf[Vector]
        )
      )

    // TODO (Brad)
    null
  }

  /*
  override def apply(rdd: RDD[(Array[(StructField, T)], Row)]): RDD[Row] = {
    val finalAttributes = rdd
      .flatMap(_._1.map(_._1).map(field => (field.name, field)))
      .reduceByKey((field, otherField) => SparkMLSupport.mergeFields(field, otherField))
      .collectAsMap()

    rdd
      .map {
        case (arr, row) => {
          val reorganized = arr.map{ case (field, value) => {
            val newField = finalAttributes(field.name)
            (newField, SparkMLSupport.reorganizeValue(field, newField, value).asInstanceOf[T])
          }}
          apply((reorganized, row))
        }
      }
  }
   */
}
