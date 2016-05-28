package com.airbnb.aerosolve.pipeline.transformers.nominal

import java.util.Locale

import com.airbnb.aerosolve.pipeline.transformers.config.PipelineConfigurable
import com.airbnb.aerosolve.pipeline.transformers.{RowTransformer, Transformer}
import com.typesafe.config.Config
import nodes.nlp.LowerCase
import com.airbnb.aerosolve.pipeline.KeystoneSupport._
import com.airbnb.aerosolve.pipeline.SparkMLSupport._

/**
 *
 */
object StringCase extends PipelineConfigurable {
  def pipeline(config : Config) = {
    val convertToUppercase = config.getBoolean("convert_to_uppercase")
    val transformer: Transformer[String, String] =
      if (convertToUppercase) new UpperCase() else LowerCase()
    RowTransformer[String](config)
      .map(transformer)
  }
}

case class UpperCase(locale: Locale = Locale.getDefault) extends Transformer[String, String] {
  override def apply(in: String): String = in.toUpperCase(locale)
}
