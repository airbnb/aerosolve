package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core.NDTreeNode
import com.airbnb.aerosolve.core.models.NDTreeModel

import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._

/*
 * Enum that describes what type of splitting algorithm should be used when constructing the tree
 */
object SplitType extends Enumeration {
  val Unspecified, Median, SurfaceArea = Value
}

/*
 * Companion object to NDTree.
 */
object NDTree {
  val splitUsingSurfaceAreaMaxCount = 10

  val defaultOneDimensionalSplitType = SplitType.Median
  val defaultMultiDimensionalSplitType = SplitType.SurfaceArea

  case class NDTreeBuildOptions(
      maxTreeDepth: Int,
      minLeafCount: Int,
      splitType: SplitType.Value = SplitType.Unspecified)

  private case class Bounds(minima: Array[Double], maxima: Array[Double])

  private case class Split(
    axisIndex: Int,
    splitValue: Double,
    leftIndices: Array[Int],
    rightIndices: Array[Int])

  def apply(options: NDTreeBuildOptions, points: Array[Array[Double]]): NDTree = {
    val nodes = ArrayBuffer[NDTreeNode]()

    val indices = points.indices.toArray

    if (indices.isEmpty) {
      return new NDTree(nodes.toArray)
    }

    val dimensions = points.head.length
    val isConstantDimension = points.forall(_.length == dimensions)

    val splitType = if (options.splitType == SplitType.Unspecified) {
      dimensions match {
        case 1 => defaultOneDimensionalSplitType
        case _ => defaultMultiDimensionalSplitType
      }
    } else {
      options.splitType
    }

    if ((indices.length > 0) && (dimensions > 0) && isConstantDimension) {
      val node = new NDTreeNode()
      nodes.append(node)
      buildTreeRecursive(options, points, indices, nodes, node, 1, splitType)
    }

    val tree = new NDTree(nodes.toArray)
    tree
  }

  private def buildTreeRecursive(
      options: NDTreeBuildOptions,
      points: Array[Array[Double]],
      indices: Array[Int],
      nodes: ArrayBuffer[NDTreeNode],
      node: NDTreeNode,
      depth: Int,
      splitType: SplitType.Value): Unit = {
    assert(indices.length > 0 && splitType != SplitType.Unspecified)

    node.setCount(indices.length)

    // Determine the min and max dimensions of the active set
    val bounds = getBounds(points, indices)
    node.setMin(bounds.minima.map(java.lang.Double.valueOf).toList.asJava)
    node.setMax(bounds.maxima.map(java.lang.Double.valueOf).toList.asJava)

    var makeLeaf = false

    if ((depth >= options.maxTreeDepth) ||
        (indices.length <= options.minLeafCount) ||
        areBoundsOverlapping(bounds)) {
      makeLeaf = true
    }

    // Recursively build tree
    if (!makeLeaf) {
      val deltas = getDeltas(bounds)

      // Choose axisIndex with largest corresponding delta
      val axisIndex = deltas.zipWithIndex.maxBy(_._1)._2

      val split = splitType match {
        case SplitType.Median =>
          getSplitUsingMedian(points, indices, axisIndex)
        case SplitType.SurfaceArea =>
          getSplitUsingSurfaceArea(points, indices, bounds, axisIndex)
      }

      if (split.leftIndices.length <= options.minLeafCount ||
          split.rightIndices.length <= options.minLeafCount) {
        makeLeaf = true
      } else {
        node.setAxisIndex(split.axisIndex)
        node.setSplitValue(split.splitValue)

        val left = new NDTreeNode()
        nodes.append(left)
        node.setLeftChild(nodes.size - 1)

        val right = new NDTreeNode()
        nodes.append(right)
        node.setRightChild(nodes.size - 1)

        buildTreeRecursive(
          options, points, split.leftIndices, nodes, left, depth + 1, splitType)
        buildTreeRecursive(
          options, points, split.rightIndices, nodes, right, depth + 1, splitType)
      }
    }

    if (makeLeaf) {
      node.setAxisIndex(NDTreeModel.LEAF)
    }
  }

  private def getBounds(
      points: Array[Array[Double]],
      indices: Array[Int]): Bounds = {
    val applicablePoints = indices.map(i => points(i))

    val minima = applicablePoints.reduceLeft((result: Array[Double], point: Array[Double]) => {
      result.zip(point).map((axisIndexValues: (Double, Double)) => {
        math.min(axisIndexValues._1, axisIndexValues._2)
      })
    })

    val maxima = applicablePoints.reduceLeft((result: Array[Double], point: Array[Double]) => {
      result.zip(point).map((axisIndexValues: (Double, Double)) => {
        math.max(axisIndexValues._1, axisIndexValues._2)
      })
    })

    Bounds(minima, maxima)
  }

  private def areBoundsOverlapping(bounds: Bounds): Boolean = {
    bounds.minima.zip(bounds.maxima).exists((bound: (Double, Double)) => {
      bound._1 >= bound._2
    })
  }

  private def getSplitUsingMedian(
      points: Array[Array[Double]],
      indices: Array[Int],
      axisIndex: Int): Split = {
    val splitValue = getMedian(points, indices, axisIndex)

    val leftIndices = indices.filter((index: Int) => {
      points(index)(axisIndex) < splitValue
    })
    val rightIndices = indices.filter((index: Int) => {
      points(index)(axisIndex) >= splitValue
    })

    val split = if ((leftIndices.length > 0) && (rightIndices.length > 0)) {
      Split(axisIndex, splitValue, leftIndices, rightIndices)
    } else {
      Split(0, 0.0, Array(), Array())
    }

    split
  }

  private def getMedian(
      points: Array[Array[Double]],
      indices: Array[Int],
      axisIndex: Int): Double = {
    // TODO (christhetree): use median of medians algorithm (quick select) for O(n)
    // performance instead of O(nln(n)) performance
    val sortedPoints = indices.map(i => points(i)).sortBy(_(axisIndex))
    val length = sortedPoints.length

    val median = if ((length % 2) == 0) {
      (sortedPoints(length / 2)(axisIndex) + sortedPoints((length / 2) - 1)(axisIndex)) / 2.0
    } else {
      sortedPoints(length / 2)(axisIndex)
    }

    median
  }

  // Split using the surface area heuristic
  // http://www.sci.utah.edu/~wald/PhD/wald_phd.pdf
  // This minimizes the cost of the split which is defined as
  // P(S(L) | S(P)) * N_L + P(S(R) | S(P)) * N_R
  // which is the surface area of the sides weighted by point counts
  private def getSplitUsingSurfaceArea(
      points: Array[Array[Double]],
      indices: Array[Int],
      bounds: Bounds,
      axisIndex: Int): Split = {
    val deltas = getDeltas(bounds)
    val parentArea = getArea(bounds)

    var bestSplit = Split(0, 0.0, Array(), Array())
    var bestScore = Double.MaxValue

    for (i <- 0 until splitUsingSurfaceAreaMaxCount) {
      val fraction = (i + 1.0) / (splitUsingSurfaceAreaMaxCount + 1.0)

      val splitValue = bounds.minima(axisIndex) + (deltas(axisIndex) * fraction)

      val leftIndices = indices.filter((index: Int) => {
        points(index)(axisIndex) < splitValue
      })
      val rightIndices = indices.filter((index: Int) => {
        points(index)(axisIndex) >= splitValue
      })

      val leftCost = computeCost(points, leftIndices, parentArea)
      val rightCost = computeCost(points, rightIndices, parentArea)

      val score = leftCost + rightCost

      if (i == 0 || ((score < bestScore) && (leftIndices.length > 0) && (rightIndices.length > 0))) {
        bestSplit = Split(axisIndex, splitValue, leftIndices, rightIndices)
        bestScore = score
      }
    }

    bestSplit
  }

  private def computeCost(
      points: Array[Array[Double]],
      indices: Array[Int],
      parentArea: Double): Double = {
    if (parentArea == 0.0) {
      return indices.length
    }

    val bounds = getBounds(points, indices)
    val area = getArea(bounds)

    if (area == 0.0) {
      return Double.MaxValue
    }

    area * indices.length.toDouble / parentArea
  }

  private def getArea(bounds: Bounds): Double = {
    val deltas = getDeltas(bounds)

    deltas.foldLeft(1.0)((area: Double, delta: Double) => {
      area * delta
    })
  }

  private def getDeltas(bounds: Bounds): Array[Double] = {
    val deltas = bounds.minima.zip(bounds.maxima).map((bound: (Double, Double)) => {
      bound._2 - bound._1
    })

    deltas
  }
}

class NDTree(val nodes: Array[NDTreeNode]) extends Serializable {
  val model = new NDTreeModel(nodes)

  // Returns the indices of nodes traversed to get to the leaf containing the point.
  def query(point: Array[Double]): Array[Int] = {
    model.query(
      point.map(_.toFloat).map(java.lang.Float.valueOf).toList.asJava
    ).asScala.map(_.intValue()).toArray
  }
//
//  // Returns the indices of all node overlapping the box
//  def queryBox(minXY : (Double, Double), maxXY : (Double, Double)) : Array[Int] = {
////    model.queryBox(minXY._1, minXY._2, maxXY._1, maxXY._2).asScala.map(x => x.intValue()).toArray
//    null
//  }

//  // Return nodes as csv.
//  // node_id, minX, minY, maxX, maxY, count, parent, is_leaf, left_child, right_child, is_xsplit, split_value
//  def getCSV() : Array[String] = {
//    val csv  = ArrayBuffer[String]()
//    getCSVRecursive(0, -1, csv)
//    return csv.toArray
//  }

//  private def getCSVRecursive(currIdx : Int, parent : Int, csv : ArrayBuffer[String]) : Unit = {
//    val builder = StringBuilder.newBuilder
//
//    val node = nodes(currIdx)
//    builder.append("%d,%f,%f,%f,%f,%d".format(
//      currIdx, node.minX, node.minY, node.maxX, node.maxY, node.count))
//    if (parent < 0) {
//      builder.append(",")
//    } else {
//      builder.append(",%d".format(parent))
//    }
//    if (node.nodeType == NDTreeNodeType.LEAF) {
//      builder.append(",TRUE,,,,")
//    } else {
//      builder.append(",FALSE,%d,%d,%s,%f".format(
//        node.leftChild,
//        node.rightChild,
//        (if (node.nodeType == NDTreeNodeType.X_SPLIT) "TRUE" else "FALSE"),
//        node.splitValue
//        ))
//    }
//    csv.append(builder.toString)
//    if (node.nodeType != NDTreeNodeType.LEAF) {
//      getCSVRecursive(node.leftChild, currIdx, csv)
//      getCSVRecursive(node.rightChild, currIdx, csv)
//    }
//  }
}
