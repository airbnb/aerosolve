package com.airbnb.aerosolve.pipeline.transformers.continuous

import com.airbnb.aerosolve.pipeline.transformers.config.{FamilyProducer, FamilySelector, FeatureVectorTransformer, PipelineConfigurable}
import com.airbnb.aerosolve.pipeline.transformers.nominal.Renamer
import com.airbnb.aerosolve.pipeline.transformers.{RowTransformer, Transformer}
import com.typesafe.config.Config
import RowTransformer._

/**
 *
 */
// QuantizeTransform
object Scaler extends PipelineConfigurable {
  def pipeline(config : Config) = {
    val selector = FamilySelector(config, "float")
      .andThen(Scaler(config))
      .andThen(Renamer("%s=%.0f", includeValue = false))
    FeatureVectorTransformer(selector, FamilyProducer(config, "string"))
  }

  def apply(config: Config): Scaler =
    Scaler(config.getDouble("scale"), floor = true)
}

case class Scaler(scale : Double, floor: Boolean = false)
  extends Transformer[Double, Double] {

  override def apply(in: Double): Double = {
    val result = in * scale
    if (floor) result.floor else result
  }
}


