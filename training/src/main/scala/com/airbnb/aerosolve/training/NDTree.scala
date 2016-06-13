package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core.NDTreeNode
import com.airbnb.aerosolve.core.models.NDTreeModel
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

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
  val log = LoggerFactory.getLogger("NDTree")
  val splitUsingSurfaceAreaMaxCount = 10

  val defaultOneDimensionalSplitType = SplitType.Median
  val defaultMultiDimensionalSplitType = SplitType.Median

  case class NDTreeBuildOptions(
    maxTreeDepth: Int,
    minLeafCount: Int,
    // (leaf's max - min)/(root's max - min)
    minLeafValuePercent: Double = 0,
    splitType: SplitType.Value = SplitType.Unspecified)

  case class Bounds(minima: Array[Double], maxima: Array[Double])

  private case class Split(
    axisIndex: Int,
    splitValue: Double,
    leftIndices: Array[Int],
    rightIndices: Array[Int])

  def buildTree(options: NDTreeBuildOptions, points: Array[Array[Double]]): Array[NDTreeNode] = {

    val indices = points.indices.toArray

    if (indices.isEmpty) {
      return Array[NDTreeNode]()
    }

    val nodes = ArrayBuffer[NDTreeNode]()
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
      buildTreeRecursive(options, points, -1, indices, nodes, node, 1, splitType)
    }
    val result = nodes.toArray
    NDTreeModel.updateWithSplitValue(result)
    result
  }

  def apply(options: NDTreeBuildOptions, points: Array[Array[Double]]): NDTree = {
    val nodes = buildTree(options, points)
    new NDTree(nodes)
  }

  def getNextAxis(lastAxis: Int,
                  deltas: Array[Double],
                  root: NDTreeNode,
                  minLeafWidthPercentage: Double): Option[Int] = {
    deltas.indices.view.map {
      i =>
        (lastAxis + i + 1) % deltas.length
    }.find{
      axis =>
        val rootWidth = root.getMax.get(axis) - root.getMin.get(axis)
        val width = deltas(axis)
        (rootWidth > 0 && width > 0 &&
          minLeafWidthPercentage * rootWidth <= width)
    }
  }

  private def buildTreeRecursive(
      options: NDTreeBuildOptions,
      points: Array[Array[Double]],
      lastAxis: Int,
      indices: Array[Int],
      nodes: ArrayBuffer[NDTreeNode],
      node: NDTreeNode,
      depth: Int,
      splitType: SplitType.Value): Unit = {
    assert(indices.length > 0 && splitType != SplitType.Unspecified)

    node.setCount(indices.length)

    // Determine the min and max dimensions of the active set
    val bounds = getBounds(points, indices)
    node.setMin(bounds.minima.map(java.lang.Double.valueOf).toBuffer.asJava)
    node.setMax(bounds.maxima.map(java.lang.Double.valueOf).toBuffer.asJava)

    var makeLeaf = false
    if ((depth >= options.maxTreeDepth) ||
        // then at least one leaf is going to smaller than minLeafCount
        (indices.length < 2 * options.minLeafCount) ||
        areBoundsOverlapping(bounds)) {
      log.debug(s"d ${depth} len ${indices.length}")
      log.debug(s"minima ${bounds.minima.mkString(",")}")
      log.debug(s"maxima ${bounds.maxima.mkString(",")}")
      makeLeaf = true
    }

    // Recursively build tree
    if (!makeLeaf) {
      val deltas = getDeltas(bounds)

      // TODO Choose axisIndex with largest corresponding delta as option: deltas.zipWithIndex.maxBy(_._1)._2
      // there is no pop in buildTreeRecursive, so nodes(0) is always the root
      val axisIndex = getNextAxis(lastAxis, deltas, nodes(0), options.minLeafValuePercent)
      if (axisIndex.isEmpty) {
        makeLeaf = true
      } else {
        val axis = axisIndex.get
        val split = splitType match {
          case SplitType.Median =>
            getSplitUsingMedian(points, indices, axis)
          case SplitType.SurfaceArea =>
            getSplitUsingSurfaceArea(points, indices, bounds, axis)
        }

        if (split.leftIndices.length < options.minLeafCount ||
          split.rightIndices.length < options.minLeafCount) {
          log.debug(s"${split.splitValue} leftIndices ${split.leftIndices.length} ${split.rightIndices.length}")
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
            options, points, axis, split.leftIndices, nodes, left, depth + 1, splitType)
          buildTreeRecursive(
            options, points, axis, split.rightIndices, nodes, right, depth + 1, splitType)
        }
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
    log.debug(s"maxima ${maxima.mkString(",")}")
    log.debug(s"applicablePoints ${applicablePoints.map(i => i.mkString(",")).mkString("|")}")

    Bounds(minima, maxima)
  }

  private def areBoundsOverlapping(bounds: Bounds): Boolean = {
    // for multi-dimension all bounds overlapping means overlapping.
    bounds.minima.zip(bounds.maxima).forall((bound: (Double, Double)) => {
      bound._1 >= bound._2
    })
  }

  private def nextGreaterElement(
      points: Array[Array[Double]], indices: Array[Int], axisIndex: Int, elem: Double): Option[Double] = {
    val data = indices.map(i => points(i)(axisIndex)).filter(i => i > elem)
    if (data.length > 0) {
      Some(data.min)
    } else {
      None
    }
  }

  private def getSplitUsingMedian(
      points: Array[Array[Double]],
      indices: Array[Int],
      axisIndex: Int): Split = {
    val splitValue = getMedian(points, indices, axisIndex)
    val (leftIndices, rightIndices) = indices.partition(index => points(index)(axisIndex) < splitValue)

    log.debug(s"getSplitUsingMedian ${splitValue} ${leftIndices.length} ${rightIndices.length}")
    if (leftIndices.length > 0) {
      // since rightIndices >= splitValue, rightIndices must have value unless points/indices empty
      Split(axisIndex, splitValue, leftIndices, rightIndices)
    } else {
      // if leftIndices empty, find next
      // median happens to be the min element
      val nextSplitValue = nextGreaterElement(points, rightIndices, axisIndex, splitValue)
      if (nextSplitValue.nonEmpty) {
        val splitValue = nextSplitValue.get
        val (left, right) = indices.partition(index => points(index)(axisIndex) < splitValue)
        Split(axisIndex, splitValue, left, right)
      } else {
        // no next greater element, so median happens to equal to both mix and max
        // this will become a leaf.
        Split(0, 0.0, Array(), Array())
      }
    }
  }

  private def getMedian(
    points: Array[Array[Double]],
    indices: Array[Int],
    axisIndex: Int): Double = {
    // TODO (christhetree): use median of medians algorithm (quick select) for O(n)
    // performance instead of O(nln(n)) performance
    // TODO use https://github.com/scalanlp/breeze/blob/master/math/src/test/scala/breeze/util/SelectTest.scala
    val sortedPoints = indices.map(i => points(i)(axisIndex)).sorted
    val length = sortedPoints.length

    val median = if ((length % 2) == 0) {
      (sortedPoints(length / 2) + sortedPoints((length / 2) - 1)) / 2.0
    } else {
      sortedPoints(length / 2)
    }

    median
  }

  // Split using the surface area heuristic
  // http://www.sci.utah.edu/~wald/PhD/wald_phd.pdf
  // This minimizes the cost of the split which is defined as
  // P(S(L) | S(P)) * N_L + P(S(R) | S(P)) * N_R
  // which is the surface area of the sides weighted by point counts
  // the idea is  to add the plane so that the surface areas of the two child spaces,
  // weighted by the number of points in each child, are equal.
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

      log.info(s"score $i $leftCost $rightCost $score $bestScore ${leftIndices.length} ${rightIndices.length}")
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
  // TODO add option to control when to use split value
  val model = new NDTreeModel(nodes)

  // Returns the indices of nodes traversed to get to the leaf containing the point.
  def query(point: Array[Double]): Array[Int] = {
    model.query(
      point.map(_.toFloat).map(java.lang.Float.valueOf).toList.asJava
    ).asScala.map(_.intValue()).toArray
  }

  // Returns the indices of all nodes overlapping the box
  def queryBox(minPoint: Array[Double], maxPoint: Array[Double]): Array[Int] = {
    model.queryBox(
      minPoint.map(java.lang.Double.valueOf).toList.asJava,
      maxPoint.map(java.lang.Double.valueOf).toList.asJava
    ).asScala.map(_.intValue()).toArray
  }

  // Return nodes as a csv:
  // nodeId, minCoordinates, maxCoordinates, count, parent, isLeaf, leftChild, rightChild,
  // axisIndex, splitValue
  def getCSV: Array[String] = {
    val csv = ArrayBuffer[String]()
    getCSVRecursive(0, -1, csv)
    csv.toArray
  }

  private def getCSVRecursive(current: Int, parent: Int, csv: ArrayBuffer[String]): Unit = {
    val builder = StringBuilder.newBuilder
    val node = nodes(current)

    builder.append("%d,".format(current))

    // TODO: instead of converting entire collection to Scala, use a Java to Scala foreach iterator
    node.min.asScala.map(_.doubleValue()).foreach((minimum: Double) => {
      builder.append("%f,".format(minimum))
    })
    node.max.asScala.map(_.doubleValue()).foreach((maximum: Double) => {
      builder.append("%f,".format(maximum))
    })

    builder.append("%d".format(node.count))

    if (parent < 0) {
      builder.append(",")
    } else {
      builder.append(",%d".format(parent))
    }

    if (node.axisIndex == NDTreeModel.LEAF) {
      builder.append(",TRUE,,,,")
    } else {
      builder.append(",FALSE,%d,%d,%d,%f".format(
        node.leftChild,
        node.rightChild,
        node.axisIndex,
        node.splitValue
      ))
    }

    csv.append(builder.toString)

    if (node.axisIndex != NDTreeModel.LEAF) {
      getCSVRecursive(node.leftChild, current, csv)
      getCSVRecursive(node.rightChild, current, csv)
    }
  }
}
