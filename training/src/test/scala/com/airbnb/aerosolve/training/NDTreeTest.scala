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
  def buildTreeUsingMedianTestWithLimitedValues: Unit = {
    val points = ArrayBuffer[Array[Double]]()

    for (x <- 1 to 100) {
      for (y <- 1 to 3) {
          points.append(Array[Double](y.toDouble))
        }
    }

    val dimensions = points.head.length

    val options = NDTreeBuildOptions(
      maxTreeDepth = 16,
      minLeafCount = 0,
      splitType = SplitType.Median)

    val tree = NDTree(options, points.toArray)
    val nodes = tree.nodes

    log.info(s"nodes = ${nodes.mkString("\n")}")
    assertEquals(5, nodes.length)
    assertEquals(2.0, nodes(0).splitValue, 0)
    assertEquals(2.5, nodes(2).splitValue, 0)
    assertEquals(1.0, nodes(1).min.get(0))
    assertEquals(1.0, nodes(1).max.get(0))
    assertEquals(3.0, nodes(4).min.get(0))
    assertEquals(3.0, nodes(4).max.get(0))
  }

  @Test
  def buildTreeUsingMedianTest: Unit = {
    val points = ArrayBuffer[Array[Double]]()

    for (x <- -2 to 6) {
      for (y <- 1 to 41) {
        for (z <- 3 to 18) {
          points.append(Array[Double](x.toDouble, y.toDouble, z.toDouble))
        }
      }
    }

    val dimensions = points.head.length

    val options = NDTreeBuildOptions(
      maxTreeDepth = 16,
      minLeafCount = 0,
      splitType = SplitType.Median)

    val tree = NDTree(options, points.toArray)
    val nodes = tree.nodes

    log.info("Number of nodes = %d".format(nodes.length))

    // Since the y dimension is largest, we expect the first node to be a split along the y axis
    assertEquals(1, nodes(0).axisIndex)
    assertEquals(21.0, nodes(0).splitValue, 0)
    assertEquals(1, nodes(0).leftChild)
    assertEquals(2, nodes(0).rightChild)

    // Ensure every point is bounded in the box of the kdtree
    for (point <- points) {
      val res = tree.query(point)
      for (index <- res) {
        for (i <- 0 until dimensions) {
          assert(point(i) >= nodes(index).min.get(i))
          assert(point(i) <= nodes(index).max.get(i))
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

  @Test
  def buildTreeUsingSurfaceAreaTest: Unit = {
    val points = ArrayBuffer[Array[Double]]()

    for (x <- -20 to 20) {
      for (y <- 1 to 19) {
        for (z <- 3 to 18) {
          points.append(Array[Double](x.toDouble, y.toDouble, z.toDouble))
        }
      }
    }

    val dimensions = points.head.length

    val options = NDTreeBuildOptions(
      maxTreeDepth = 16,
      minLeafCount = 0,
      splitType = SplitType.SurfaceArea)

    val tree = NDTree(options, points.toArray)
    val nodes = tree.nodes

    log.info("Number of nodes = %d".format(nodes.length))

    // Since the x dimension is largest, we expect the first node to be a split along the x axis
    assertEquals(0, nodes(0).axisIndex)
    assertEquals(-1.81, nodes(0).splitValue, 0.1)
    assertEquals(1, nodes(0).leftChild)
    assertEquals(2, nodes(0).rightChild)

    // Ensure every point is bounded in the box of the kdtree
    for (point <- points) {
      val res = tree.query(point)
      for (index <- res) {
        for (i <- 0 until dimensions) {
          assert(point(i) >= nodes(index).min.get(i))
          assert(point(i) <= nodes(index).max.get(i))
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
