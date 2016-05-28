package com.airbnb.aerosolve.pipeline.transformers

import com.airbnb.aerosolve.pipeline.transformers.continuous.Scaler
import com.airbnb.aerosolve.pipeline.transformers.nominal.StringCase
import com.typesafe.config.Config
import org.apache.spark.sql.Row

/**
 *
 */
object Transforms {

  def pipelineConfigurer(name: String) = name match {
    case "scale" => Scaler
    case "quantize" => Scaler
    case "convert_string_case" => StringCase
    case "string_case" => StringCase
  }

  def pipeline(config: Config, key: String): Option[Transformer[Row, Row]] =
    (config, key) match {
      case (null, _) => None
      case (_, null) => None
      case _ if !config.hasPath(s"$key.transform") => None
      case _ => {
        val clazz = pipelineConfigurer(config.getString(s"$key.transform"))
        Some(clazz.pipeline(config.getConfig(key)))
      }
    }
}
