package com.airbnb.aerosolve.pipeline.transformers

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.json4s.jackson.Serialization

import scala.reflect.ClassTag

/**
 *
 */
object Transformer {
  def identity[I: ClassTag] = new Identity[I]
}

abstract class Transformer[-I, +O: ClassTag] extends (I => O) with JsonWritable {

  implicit val formats = Serialization.formats(new TransformerTypeHints())

  def apply(rdd: RDD[_ <: I]): RDD[_ <: O] = {
    rdd.map(apply)
  }

  def andThen[M:ClassTag](t: Transformer[O, M]): this.type = {
    Pipeline(this, t)
  }

  override def toJson = Serialization.write(this)

  def apply(input: I): O
}

trait JsonWritable {
  def toJson: String
}

case class Pipeline[-I, M, +O: ClassTag](left: Transformer[_ >: I, _ <: M],
                                         right: Transformer[_ >: M, _ <: O])
  extends Transformer[I, O] {

  override def apply(v1: I): O = right(left(v1))
}

class Identity[I:ClassTag] extends Transformer[I, I] {

  override def apply(v1: I): I = v1
}