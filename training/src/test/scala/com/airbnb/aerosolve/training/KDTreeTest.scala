package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core.KDTreeNode
import com.airbnb.aerosolve.core.KDTreeNodeType
import com.airbnb.aerosolve.training.KDTree.KDTreeBuildOptions
import org.junit.Test
import org.slf4j.LoggerFactory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import scala.collection.mutable.ArrayBuffer

class KDTreeTest {
  val log = LoggerFactory.getLogger("KDTreeTest")

  @Test def buildTreeTest: Unit = {
    val pts = ArrayBuffer[(Double, Double)]()
    for (x <- -20 to 20) {
      for (y <- 1 to 5) {
        pts.append((x.toDouble, y.toDouble))
      }
    }
    val options = KDTreeBuildOptions(maxTreeDepth = 16, minLeafCount = 1)
    val tree = KDTree(options, pts.toArray)
    val nodes = tree.nodes
    log.info("Num nodes = %d".format(nodes.size))
    // Since the x dimension is largest we expect the first node to be an xsplit
    assertEquals(KDTreeNodeType.X_SPLIT, nodes(0).nodeType)
    assertEquals(-1.81, nodes(0).splitValue, 0.1)
    assertEquals(1, nodes(0).leftChild)
    assertEquals(2, nodes(0).rightChild)    // Ensure every point is bounded in the box of the kdtree
    for (pt <- pts) {
      val res = tree.query(pt)
      for (idx <- res) {
        assert(pt._1 >= nodes(idx).minX)
        assert(pt._1 <= nodes(idx).maxX)
        assert(pt._2 >= nodes(idx).minY)
        assert(pt._2 <= nodes(idx).maxY)
      }
    }
    // Ensure all nodes are sensible
    for (node <- nodes) {
      assert(node.count > 0)
      assert(node.minX <= node.maxX)
      assert(node.minY <= node.maxY)
      if (node.nodeType != KDTreeNodeType.LEAF) {
        assert(node.leftChild >= 0 && node.leftChild < nodes.size)
        assert(node.rightChild >= 0 && node.rightChild < nodes.size)
      }
    }
  }

}