package com.airbnb.aerosolve.training

import org.slf4j.{LoggerFactory, Logger}
import scala.math

/*
 * A trainer that uses cyclic coordinate descent to compute the minimum of a function
 */

object CyclicCoordinateDescent {
  case class Params(tolerance : Double,
                    iterations : Int,
                    initial : Array[Double],
                    initialStep : Array[Double],
                    bounds : Array[(Double, Double)])

  // Given function f, initial location and bounds on the search
  // return the location that best minimizes f up to tolerance precision
  def optimize(f : Array[Double] => Double,
               params : Params) : Array[Double]  = {
    var best : Array[Double] = params.initial
    var bestF : Double = f(best)
    val maxDim = params.initial.length
    for (it <- 1 to params.iterations) {
      for (dim <- 0 until maxDim) {
        var step = params.initialStep(dim)
        while (step > params.tolerance) {
          val left = best.clone()
          left(dim) = math.max(params.bounds(dim)._1, best(dim) - step)
          val leftF = f(left)
          val right = best.clone()
          right(dim) = math.min(params.bounds(dim)._2, best(dim) + step)
          val rightF = f(right)
          if (leftF < bestF) {
            best = left
            bestF = leftF
          }
          if (rightF < bestF) {
            best = right
            bestF = rightF
          }
          step = step * 0.5
        }
      }
    }

    return best
  }
}
