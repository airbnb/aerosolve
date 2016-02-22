package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core.util.FloatVector
import org.slf4j.{LoggerFactory, Logger}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.util.Random

/*
 * A helper object that has re-usable utilities for dealing with gradients
 *
 * */

object GradientUtils {
  private final val log: Logger = LoggerFactory.getLogger("GradientUtils")

  case class GradientContainer(grad : FloatVector, featureSquaredSum : Double)

  def sumGradients(a : GradientContainer, b : GradientContainer) : GradientContainer = {
    a.grad.add(b.grad)
    GradientContainer(a.grad, a.featureSquaredSum + b.featureSquaredSum)
  }

  // Gradient update rule from "boosting with structural sparsity Duchi et al 2009"
  def sparseBoost(gradients : Map[(String, String), GradientContainer],
                  weightVector : java.util.Map[String,java.util.Map[String,com.airbnb.aerosolve.core.util.FloatVector]],
                  dim :Int,
                  lambda : Double) = {
    var gradientNorm = 0.0
    var featureCount = 0
    gradients.foreach(kv => {
      val (key, gradient) = kv
      val featureMap = weightVector.get(key._1)
      if (featureMap != null) {
        val weight = featureMap.get(key._2)
        if (weight != null) {
          // Just a proxy measure for convergence.
          gradientNorm = gradientNorm + gradient.grad.dot(gradient.grad)
          val scale = 2.0 / math.max(1e-6, gradient.featureSquaredSum)
          weight.multiplyAdd(-scale.toFloat, gradient.grad)
          val hingeScale = 1.0 - lambda * scale / math.sqrt(weight.dot(weight))
          if (hingeScale <= 0.0f) {
            // Weights may get re-activated ... filter at the end.
            weight.setZero(dim)
          } else {
            featureCount = featureCount + 1
            weight.scale(hingeScale.toFloat)
          }
        }
      }
    })
    log.info("Sum of Gradient L2 norms = " + gradientNorm)
    log.info("Num active features = " + featureCount)
  }

  // Improved RPROP- algorithm
  // https://en.wikipedia.org/wiki/Rprop
  // http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.3428
  def rprop(gradients : Map[(String, String), GradientContainer],
            prevGradients : Map[(String, String), GradientContainer],
            step : scala.collection.mutable.HashMap[(String, String), FloatVector],
            weightVector : java.util.Map[String,java.util.Map[String,com.airbnb.aerosolve.core.util.FloatVector]],
            dim :Int,
            lambda : Double,
            deltaMax : Float = 1.0f) = {
    // Some fixed parameters for rprop, they don't matter much as the algorithm adapts them
    // so just keep them fixed here.
    // Step size parameters
    val deltaInitial = 0.01f
    val deltaMin = 1e-6f
    // Change in step size parameters
    val etaPlus = 1.2f
    val etaMinus = 0.5f

    gradients.foreach(kv => {
      val (key, gradient) = kv
      val featureMap = weightVector.get(key._1)
      if (featureMap != null) {
        val weight = featureMap.get(key._2)
        if (weight != null) {
          val prev = prevGradients.get(key)
          val prevGrad = if (prev.isEmpty) new FloatVector(dim) else prev.get.grad
          val currOpt = step.get(key)
          // Create a new step vector if we don't have one.
          if (currOpt.isEmpty) {
            val tmp = new FloatVector(dim)
            tmp.setConstant(deltaInitial)
            step.put(key, tmp)
          }
          val currStep = step.get(key).get
          for (i <- 0 until dim) {
            // L2 regularization term
            gradient.grad.values(i) = gradient.grad.values(i) + lambda.toFloat * weight.values(i)

            val prod = prevGrad.values(i) * gradient.grad.values(i)
            if (prod > 0) {
              currStep.values(i) = math.min(currStep.values(i) * etaPlus, deltaMax)
            } else if (prod < 0) {
              currStep.values(i) = math.max(currStep.values(i) * etaMinus, deltaMin)
              gradient.grad.values(i) = 0
            }
            val sign = if (gradient.grad.values(i) > 0) 1.0f else -1.0f
            weight.values(i) = weight.values(i) - sign * currStep.values(i)

          }
        }
      }
    })
    val stepNorms = step.map(kv => math.sqrt(kv._2.dot(kv._2)))
    log.info("Sum of Step L2 norms = " + stepNorms.sum)
    log.info("Average of Step L2 norms = " + stepNorms.sum / stepNorms.size)
    log.info("Max of Step L2 norms = " + stepNorms.max)
    val median = stepNorms.toBuffer.sorted.get(stepNorms.size / 2)
    log.info("Median Step L2 norms = " + median)
  }

  def gradientDescent(gradients : Map[(String, String), GradientContainer],
                      weightVector : java.util.Map[String,java.util.Map[String,com.airbnb.aerosolve.core.util.FloatVector]],
                      dim :Int,
                      learningRate: Double,
                      lambda : Double) = {
    gradients.foreach(kv => {
      val (key, gradient) = kv
      val featureMap = weightVector.get(key._1)
      if (featureMap != null) {
        val weight: FloatVector = featureMap.get(key._2)
        if (weight != null) {
          for (i <- 0 until dim) {
            // L2 regularization term determined by lambda
            gradient.grad.values(i) = gradient.grad.values(i) + lambda.toFloat * weight.values(i)
            weight.values(i) -= learningRate.toFloat * gradient.grad.values(i)
          }
        }
      }
    })
  }
}
