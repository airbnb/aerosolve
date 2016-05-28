package com.airbnb.aerosolve.pipeline.transformers.select

import com.airbnb.aerosolve.pipeline.transformers.Transformer
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.sql.Row

/**
 *
 */
case class VectorSelector(column: String,
                          mapper: Option[Transformer[Double, Double]] = None)
  extends Transformer[Row, Vector] {

  override def apply(row: Row): Vector = {
    val vector = row.get(row.fieldIndex(column)).asInstanceOf[Vector]

    mapper.map(t => {
      val doubles = Array[Double](vector.numActives)
      val indexes = Array[Int](vector.numActives)
      var i = 0
      vector.foreachActive((index, value) => {
        doubles(i) = t(value)
        indexes(i) = index
        i += 1
      })
      Vectors.sparse(indexes.length, indexes, doubles)
    }).getOrElse(vector)
  }

  def map(transformer: Transformer[Double, Double]): Transformer[Row, Vector] = {
    VectorSelector(column, mapper.map(_.andThen(transformer)).orElse(Some(transformer)))
  }
}
