package com.airbnb.aerosolve.pipeline.transformers.config

import com.airbnb.aerosolve.pipeline.transformers.Transformer
import com.typesafe.config.Config

/**
 *
 */
trait Configurable[T <: Transformer[_, _]] {
  def apply(config: Config): T
}
