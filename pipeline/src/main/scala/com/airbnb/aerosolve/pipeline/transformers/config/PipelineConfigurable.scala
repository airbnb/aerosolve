package com.airbnb.aerosolve.pipeline.transformers.config

import com.airbnb.aerosolve.pipeline.transformers.Transformer
import com.typesafe.config.Config
import org.apache.spark.sql.Row

/**
 *
 */
trait PipelineConfigurable {
  def pipeline(config : Config) : Transformer[Row, Row]
}
