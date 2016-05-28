package com.airbnb.aerosolve.pipeline.transformers.config

import com.airbnb.aerosolve.core.FeatureVector
import com.airbnb.aerosolve.pipeline.transformers.Transformer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

/**
 *
 */
case class FeatureVectorTransformer(selector: Transformer[FeatureVector, Row],
                                    producer: Transformer[(Row, FeatureVector), FeatureVector])
  extends Transformer[FeatureVector, FeatureVector]{

  override def apply(rdd: RDD[FeatureVector]): RDD[FeatureVector] = {
    producer(selector(rdd).zip(rdd))
  }

  override def apply(vector: FeatureVector): FeatureVector = {
    producer(selector(vector), vector)
  }
}
