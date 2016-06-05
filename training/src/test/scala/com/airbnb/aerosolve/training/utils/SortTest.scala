package com.airbnb.aerosolve.training.utils

import org.junit.Assert._
import org.junit.Test
import org.slf4j.LoggerFactory

class SortTest {
  val log = LoggerFactory.getLogger("SortTest")
  @Test
  def testQuickSelect : Unit = {
    val v = Array(9, 8, 7, 6, 5, 0, 1, 2, 3, 4)
    val r = (0 until v.length).map(Sort.quickSelect(v, _)).toArray
    assertTrue(Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9).deep == r.deep)

    val v2 = Array(Array(9), Array(8), Array(7), Array(6), Array(5), Array(0), Array(1), Array(2), Array(3), Array(4))

    val r2 = (0 until v2.length).map(Sort.quickSelectAxis[Int](v2, _, 0)).toArray
    assertTrue(Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9).deep == r2.deep)

    assertEquals(1, Sort.quickSelect(Array(1,1,1,1,1), 0))
    assertEquals(1, Sort.quickSelect(Array(1,1,1,1,1), 1))
    assertEquals(1, Sort.quickSelect(Array(1,1,1,2,2,2), 2))
    assertEquals(2, Sort.quickSelect(Array(1,1,1,2,2,2), 3))
    assertEquals(1, Sort.quickSelect(Array(0,1,2,1), 2))
    assertEquals(0, Sort.quickSelect(Array(0,0,0,0,1,2,1), 3))

  }
}
