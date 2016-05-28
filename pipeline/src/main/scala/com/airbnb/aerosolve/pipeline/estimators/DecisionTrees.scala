package com.airbnb.aerosolve.pipeline.estimators

import org.apache.spark.{ml => ml}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.ml.tree.{InternalNode, LeafNode, Node}
import org.json4s.jackson.JsonMethods._
import org.json4s._
import workflow.Transformer

import scala.reflect.io.File

/**
 *
 */
object DecisionTrees {
  implicit val formats = DefaultFormats

  def load(fileName: String): DecisionTreeClassificationModel = {
    parse(File(fileName).slurp()).extract[DecisionTreeClassificationModel]
  }

  implicit def apply(m: ml.Model[_]):DecisionTreeModel = {
    m match {
      case model: ml.classification.DecisionTreeClassificationModel =>
        DecisionTreeClassificationModel(DecisionTreeNode(model.rootNode), model.numClasses)
      case model: ml.regression.DecisionTreeRegressionModel =>
        DecisionTreeRegressionModel(DecisionTreeNode(model.rootNode))
    }
  }

  /* implicit def apply(model: ml.regression.DecisionTreeRegressionModel):
    DecisionTreeRegressionModel = {

  }*/

  case class DecisionTreeClassificationModel(node: DecisionTreeNode, numClasses: Int)
    extends Transformer[Vector, Array[Double]] with DecisionTreeModel {

    override def apply(in: Vector): Array[Double] = {
      // TODO (Brad)
      Array[Double](1.0)
    }
  }

  case class DecisionTreeRegressionModel(node: DecisionTreeNode)
    extends Transformer[Vector, Array[Double]] with DecisionTreeModel {

    override def apply(in: Vector): Array[Double] = {
      // TODO (Brad)
      Array[Double](1.0)
    }
  }

  trait DecisionTreeModel {}
}

object DecisionTreeNode {
  def apply(node: Node): DecisionTreeNode = {
    node match {
      case n: LeafNode => null // TODO
      case n: InternalNode => null // TODO
    }
  }
}

trait DecisionTreeNode {}

case class InternalDecisionTreeNode(prediction: Double, impurity: Double, gain: Double,
                                leftChild: DecisionTreeNode, rightChild:DecisionTreeNode,
                                split: Split) extends DecisionTreeNode{
}

// TODO (Brad)
case class Split(a:String)
