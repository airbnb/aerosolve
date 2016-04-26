package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core.models.NDTreeModel
import com.airbnb.aerosolve.training.NDTree.NDTreeBuildOptions
import org.junit.Assert.assertEquals
import org.junit.Test
import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer

class NDTreeTest {
  val log = LoggerFactory.getLogger("NDTreeTest")

  @Test
  def buildTreeTest: Unit = {
    val pts = ArrayBuffer[List[Double]]()
    for (x <- -20 to 20) {
      for (y <- 1 to 5) {
        for (z <- 3 to 4) {
          pts.append(List[Double](x.toDouble, y.toDouble, z.toDouble))
        }
      }
    }
    val dimensions = pts.head.length
    val options = NDTreeBuildOptions(maxTreeDepth = 16, minLeafCount = 1)
    val tree = NDTree(options, pts.toArray)
    val nodes = tree.nodes
    log.info("Num nodes = %d".format(nodes.size))
    // Since the x dimension is largest we expect the first node to be an xsplit
    assertEquals(0, nodes(0).axisIndex)
    assertEquals(-1.81, nodes(0).splitValue, 0.1)
    assertEquals(1, nodes(0).leftChild)
    assertEquals(2, nodes(0).rightChild)    // Ensure every point is bounded in the box of the kdtree
    for (pt <- pts) {
      val res = tree.query(pt)
      for (idx <- res) {
        for (i <- 0 until dimensions) {
          assert(pt(i) >= nodes(idx).min.get(i))
          assert(pt(i) <= nodes(idx).max.get(i))
        }
      }
    }
    // Ensure all nodes are sensible
    for (node <- nodes) {
      assert(node.count > 0)

      for (i <- 0 until dimensions) {
        assert(node.min.get(i) <= node.max.get(i))
      }
      if (node.axisIndex != NDTreeModel.LEAF) {
        assert(node.leftChild >= 0 && node.leftChild < nodes.length)
        assert(node.rightChild >= 0 && node.rightChild < nodes.length)
      }
    }
  }
}
