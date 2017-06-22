package com.airbnb.common.ml.util

import scala.util.Random


object RandomUtil {
  def randomDouble(bounds: Seq[Any], randomizer: Random): Double = {
    val min = bounds.head.asInstanceOf[Double]
    val max = bounds.tail.head.asInstanceOf[Double]
    (randomizer.nextDouble * (max - min)) + min
  }

  def randomInt(bounds: Seq[Any], randomizer: Random): Int = {
    val min = bounds.head.asInstanceOf[Int]
    val max = bounds.tail.head.asInstanceOf[Int]
    randomizer.nextInt(max - min) + min
  }

  def randomNumber(bounds: Seq[Any], randomizer: Random): Any = {
    if (bounds.head.isInstanceOf[Int]) {
      randomInt(bounds, randomizer)
    } else {
      randomDouble(bounds, randomizer)
    }
  }

  def randomIndex(bounds: Seq[Any], randomizer: Random): Any = {
    val index = randomizer.nextInt(bounds.length)
    bounds(index)
  }

  def sample[T](items: Seq[T], ratios: Seq[Double]): Seq[Seq[T]] = {
    val shuffledItems: Seq[T] = Random.shuffle(items)
    slice(shuffledItems, ratios)
  }

  def slice[T](items: Seq[T], ratios: Seq[Double]): Seq[Seq[T]] = {
    val itemCount: Int = items.length
    val startPositions: Seq[Double] = ratios.scanLeft(0.0)(_ + _).take(ratios.length)

    // The range here is ratio-based, which is converted to index
    // through multiplication with itemCount.
    startPositions.zip(ratios).map {
      case (rangeStart: Double, rangeLength: Double) => {
        val startPos: Int = (rangeStart * itemCount).toInt
        val endPos: Int = ((rangeStart + rangeLength) * itemCount).toInt
        items.slice(startPos, endPos)
      }
    }
  }
}
