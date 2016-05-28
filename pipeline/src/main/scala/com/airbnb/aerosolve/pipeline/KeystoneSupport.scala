package com.airbnb.aerosolve.pipeline

import com.airbnb.aerosolve.pipeline.transformers.Transformer

import scala.reflect.ClassTag

/**
 *
 */
object KeystoneSupport {

  implicit class KeystoneTransformer[I, O: ClassTag](t: workflow.Transformer[I, O])
    extends Transformer[I, O] {
    override def apply(in: I): O = t.apply(in)

    // TODO
    override def toJson: String = ""
  }

}
