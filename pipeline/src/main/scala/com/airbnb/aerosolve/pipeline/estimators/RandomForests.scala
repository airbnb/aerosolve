package com.airbnb.aerosolve.pipeline.estimators

import com.airbnb.aerosolve.pipeline.transformers.Transformer
import org.apache.spark.ml
import org.apache.spark.ml.{classification => spark, Model}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import com.airbnb.aerosolve.pipeline.estimators.DecisionTrees.{DecisionTreeModel, DecisionTreeClassificationModel}

/**
 *
 */
object RandomForestClassificationModel {
  implicit def apply(model: spark.RandomForestClassificationModel): RandomForestClassificationModel = {
    RandomForestClassificationModel(model.trees.map(t => DecisionTrees(t.asInstanceOf[Model[_]])),
                                    model.numFeatures,
                                    model.numClasses)
  }
}

case class RandomForestClassificationModel(trees: Array[DecisionTreeModel],
                                           numFeatures: Int,
                                           numClasses: Int)
  extends Transformer[Vector, Vector] {

  override def apply(in: Vector): Vector = {
    val votes = Array.fill[Double](numClasses)(0.0)
    trees.foreach { tree =>
      val classCounts: Array[Double] = tree match {
        case tree: DecisionTreeClassificationModel => tree.apply(in)
      }
      val total = classCounts.sum
      if (total != 0) {
        var i = 0
        while (i < numClasses) {
          votes(i) += classCounts(i) / total
          i += 1
        }
      }
    }
    Vectors.dense(votes)
  }
}

