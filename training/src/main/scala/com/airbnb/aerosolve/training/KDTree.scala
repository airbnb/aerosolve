package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core.{KDTreeNode, KDTreeNodeType}
import com.airbnb.aerosolve.core.models.KDTreeModel

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

// A specialized 2D Kdtree that supports accumulating boxes of values.
class KDTree(val nodes: Array[KDTreeNode]) extends Serializable {
  val model = new KDTreeModel(nodes)

  // Returns the indices of nodes traversed to get to the leaf containing the point.
  def query(loc: (Double, Double)): Array[Int] = {
    model.query(loc._1, loc._2).asScala.map(x => x.intValue()).toArray
  }

  // Returns the indices of all node overlapping the box
  def queryBox(minXY: (Double, Double), maxXY: (Double, Double)): Array[Int] = {
    model.queryBox(minXY._1, minXY._2, maxXY._1, maxXY._2).asScala.map(x => x.intValue()).toArray
  }

  // Return nodes as csv.
  // node_id, minX, minY, maxX, maxY, count, parent, is_leaf, left_child, right_child, is_xsplit, split_value
  def getCSV(): Array[String] = {
    val csv = ArrayBuffer[String]()
    getCSVRecursive(0, -1, csv)
    csv.toArray
  }

  private def getCSVRecursive(currIdx: Int, parent: Int, csv: ArrayBuffer[String]): Unit = {
    val builder = StringBuilder.newBuilder

    val node = nodes(currIdx)
    builder.append("%d,%f,%f,%f,%f,%d".format(
      currIdx, node.minX, node.minY, node.maxX, node.maxY, node.count))
    if (parent < 0) {
      builder.append(",")
    } else {
      builder.append(",%d".format(parent))
    }
    if (node.nodeType == KDTreeNodeType.LEAF) {
      builder.append(",TRUE,,,,")
    } else {
      builder.append(",FALSE,%d,%d,%s,%f".format(
        node.leftChild,
        node.rightChild,
        if (node.nodeType == KDTreeNodeType.X_SPLIT) "TRUE" else "FALSE",
        node.splitValue
      ))
    }
    csv.append(builder.toString)

    if (node.nodeType != KDTreeNodeType.LEAF) {
      getCSVRecursive(node.leftChild, currIdx, csv)
      getCSVRecursive(node.rightChild, currIdx, csv)
    }
  }

  /**
    * Update bounds of KDTree node in-place
    *
    * @note bounding box is informational and is not used in pinpoint observation
    *       into a KDTree node. This information was used in KDTree split cost computation.
    * @param bounds a map from KDTree node id to bounding box
    */
  def updateBounds(bounds: Map[Int, KDTree.Bounds]): Unit = {
    nodes.iterator.zipWithIndex.foreach {
      case (node, id) =>
        bounds.get(id)
          .foreach {
            bound =>
              node.setMinX(bound.minX)
              node.setMaxX(bound.maxX)
              node.setMinY(bound.minY)
              node.setMaxY(bound.maxY)
          }
    }
  }
}

object KDTree {

  case class KDTreeBuildOptions(maxTreeDepth: Int, minLeafCount: Int)

  def apply(options: KDTreeBuildOptions,
            pts: Array[(Double, Double)]): KDTree = {
    val nodes = ArrayBuffer[KDTreeNode]()
    val idx: Array[Int] = pts.indices.toArray
    if (idx.length > 0) {
      val node = new KDTreeNode()
      nodes.append(node)
      buildTreeRecursive(options, pts, idx, nodes, node, 1)
    }
    new KDTree(nodes.toArray)
  }

  case class Bounds(minX: Double, maxX: Double, minY: Double, maxY: Double)

  case class Split(xSplit: Boolean, splitVal: Double, leftIdx: Array[Int], rightIdx: Array[Int])

  private def buildTreeRecursive(options: KDTreeBuildOptions,
                                 pts: Array[(Double, Double)],
                                 idx: Array[Int],
                                 nodes: ArrayBuffer[KDTreeNode],
                                 node: KDTreeNode,
                                 depth: Int): Unit = {
    assert(idx.length > 0)
    node.setCount(idx.length)
    // Determine the min and max dimensions of the active set
    // Active points are triplets of x, y and index
    val bounds = getBounds(pts, idx)
    node.setMinX(bounds.minX)
    node.setMaxX(bounds.maxX)
    node.setMinY(bounds.minY)
    node.setMaxY(bounds.maxY)

    var makeLeaf: Boolean = false
    if (depth >= options.maxTreeDepth ||
      idx.length <= options.minLeafCount ||
      (bounds.minX >= bounds.maxX) ||
      (bounds.minY >= bounds.maxY)) {
      makeLeaf = true
    }

    // Recursively build tree
    if (!makeLeaf) {
      val split = getSplitVal(pts, idx, bounds)
      if (split.leftIdx.length <= options.minLeafCount || split.rightIdx.length <= options.minLeafCount) {
        makeLeaf = true
      } else {
        val nodeType = if (split.xSplit) KDTreeNodeType.X_SPLIT else KDTreeNodeType.Y_SPLIT
        node.setNodeType(nodeType)
        node.setSplitValue(split.splitVal)
        val left = new KDTreeNode()
        nodes.append(left)
        node.setLeftChild(nodes.size - 1)
        val right = new KDTreeNode()
        nodes.append(right)
        node.setRightChild(nodes.size - 1)
        buildTreeRecursive(options, pts, split.leftIdx, nodes, left, depth + 1)
        buildTreeRecursive(options, pts, split.rightIdx, nodes, right, depth + 1)
      }
    }
    if (makeLeaf) {
      node.setNodeType(KDTreeNodeType.LEAF)
    }
  }

  private def getBounds(pts: Array[(Double, Double)],
                        idx: Array[Int]): Bounds = {
    val res = idx.map(i => pts(i))
      .map(pt => Bounds(pt._1, pt._1, pt._2, pt._2))
    res.fold(res.head) { (a, b) =>
      Bounds(math.min(a.minX, b.minX), math.max(a.maxX, b.maxX),
        math.min(a.minY, b.minY), math.max(a.maxY, b.maxY))
    }

  }

  private def getArea(bounds: Bounds): Double = {
    (bounds.maxX - bounds.minX) * (bounds.maxY - bounds.minY)
  }

  // Split using the surface area heuristic
  // http://www.sci.utah.edu/~wald/PhD/wald_phd.pdf
  // This minimizes the cost of the split which is defined as
  // P(S(L) | S(P)) * N_L + P(S(R) | S(P)) * N_R
  // which is the surface area of the sides weighted by point counts
  private def computeCost(pts: Array[(Double, Double)],
                          idx: Array[Int],
                          parentArea: Double): Double = {
    if (parentArea == 0.0) {
      idx.length
    } else {
      val bounds = getBounds(pts, idx)
      val area = getArea(bounds)
      if (area == 0.0) {
        Double.MaxValue
      } else {
        area * idx.length.toDouble / parentArea
      }
    }
  }

  private def getSplitVal(pts: Array[(Double, Double)],
                          idx: Array[Int],
                          bounds: Bounds): Split = {
    val dx = bounds.maxX - bounds.minX
    val dy = bounds.maxY - bounds.minY
    val parentArea = dx * dy
    val xSplit = dx > dy

    var bestSplit: Split = Split(true, 0.0, Array(), Array())
    var bestScore: Double = Double.MaxValue

    // Sample uniformly for the best split
    val MAX_COUNT: Int = 10
    for (i <- 0 until MAX_COUNT) {
      val frac: Double = (i + 1.0) / (MAX_COUNT + 1.0)
      val splitVal = if (xSplit) bounds.minX + dx * frac else bounds.minY + dy * frac

      val leftIdx = if (xSplit) {
        idx.filter(id => pts(id)._1 < splitVal)
      } else {
        idx.filter(id => pts(id)._2 < splitVal)
      }

      val rightIdx = if (xSplit) {
        idx.filter(id => pts(id)._1 >= splitVal)
      } else {
        idx.filter(id => pts(id)._2 >= splitVal)
      }

      val leftCost = computeCost(pts, leftIdx, parentArea)
      val rightCost = computeCost(pts, rightIdx, parentArea)
      val score = leftCost + rightCost
      if (i == 0 || (score < bestScore && leftIdx.length > 0 && rightIdx.length > 0)) {
        bestSplit = Split(xSplit, splitVal, leftIdx, rightIdx)
        bestScore = score
      }
    }

    bestSplit
  }
}
