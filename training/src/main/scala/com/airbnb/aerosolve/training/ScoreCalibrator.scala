package com.airbnb.aerosolve.training

import com.typesafe.config.Config
import org.slf4j.{Logger, LoggerFactory}
import org.apache.spark.rdd.RDD
import scala.util.Random
/*
* A logistic regression trainer for generating Platt's scaling weights (slope and offset)
* The objective function to minimize is the negative log-likelihood function.
* 'trainBatchMLE' implements batch gradient descent, which collects data from RDD and run training on local machine
* 'trainSGD' implements Stochastic Gradient Descent (SGD), which runs in parallel model.
* Params:
* learning_rate - a number between 0 and 1.
* iterations - maximal number of iterations to run.
* num_bags - number of parallel models to make (sgd only).
* tolerance - if the difference between previous iteration weights and current weights are
*             smaller than tolerance, the training will stop.
* rate_decay - reduce learning rate: learning_rate = learning_rate * rate_decay after each epoch
*/

object ScoreCalibrator {
  private final val log: Logger = LoggerFactory.getLogger("ScoreCalibrator")
  def sigmoid(x: Double) = 1.0 / (1.0 + math.exp(-x))

  def trainSGD(config : Config,
               input : RDD[(Double, Boolean)]) : Array[Double] = {
    /* Config info:
    * iterations - maximal number of iterations to run trainMiniSGD
    * numBags - number of parallel model to train
    * learning_rate - learning rate
    * rate_decay - reduce learning rate: learning_rate = learning_rate * rate_decay after each epoch
    * tolerance - if the difference between previous iteration weights and current weights are
    *             smaller than tolerance, the training will stop.
    */
    val maxIter : Int = config.getInt("iterations")
    var learningRate : Double = config.getDouble("learning_rate")
    val numBags : Int = config.getInt("num_bags")
    val tolerance : Double = config.getDouble("tolerance")
    val rate_decay : Double = config.getDouble("rate_decay")
    var params = Array(0.0, 1.0)
    var iter = 0
    var old_offset = 0.0
    var old_slope = 1.0
    val partitionedInput = input.repartition(numBags).cache()

    do {
      old_offset = params(0)
      old_slope = params(1)
      params = trainMiniSGD(partitionedInput, numBags, learningRate, params(0), params(1))
      iter += 1
      learningRate = learningRate * rate_decay
      log.info("Calibration Iteration %d: offset = %f, slope = %f".format(iter, params(0), params(1)))
    } while((math.abs(old_offset - params(0)) > tolerance
      || math.abs(old_slope - params(1)) > tolerance)
      && iter <= maxIter)

    partitionedInput.unpersist()

    log.info("Calibration stopped at iteration %d".format(iter))
    params
  }

  def trainMiniBatch(input : RDD[(Double, Boolean)],
                     numBags : Int,
                     learningRate : Double,
                     offset : Double,
                     slope : Double) : Array[Double] = {
    val result = input
      .mapPartitions(partition => {
      // run mini-batch training on each partition
      var a = offset
      var b = slope
      var count = 0
      var gradientA = 0.0
      var gradientB = 0.0
      partition.foreach(x => { // x is (Double, Boolean)
      // compute gradient at a given data point
        val diff = sigmoid(a + b * x._1) - (if (x._2) 1.0 else 0.0)
        gradientA += diff
        gradientB += diff * x._1
        count += 1
      })
      a -= learningRate * gradientA / count
      b -= learningRate * gradientB / count
      Seq[(Double, Double)]((a, b)).iterator
    })
      .reduce((x, y) => (x._1 + y._1, x._2 + y._2))

    Array(result._1 / numBags, result._2 / numBags)
  }

  def trainMiniSGD(input : RDD[(Double, Boolean)],
                   numBags : Int,
                   learningRate : Double,
                   offset : Double,
                   slope : Double) : Array[Double] = {
    // First shuffle the data otherwise SGD won't work.
    val data = input.map(x => (math.random, x))
      .sortBy(_._1, numPartitions = numBags).values
    val result = data
      .mapPartitions(partition => {
      // run mini-batch training on each partition
      var a = offset
      var b = slope
      partition.foreach(x => { // x is (Double, Boolean)
      // compute gradient at a given data point
        val diff = sigmoid(a + b * x._1) - (if (x._2) 1.0 else 0.0)
        a -= learningRate * diff
        b -= learningRate * diff * x._1
      })
      Seq[(Double, Double)]((a, b)).iterator
    })
      .reduce((x, y) => (x._1 + y._1, x._2 + y._2))

    Array(result._1 / numBags, result._2 / numBags)
  }

  // Batch Gradient Descent: Maximum Likelihood Estimate (MLE)
  def trainBatchMLE(config : Config,
                    input : Array[(Double, Boolean)]) : Array[Double] = {

    val maxIter : Int = config.getInt("iterations")
    var learningRate : Double = config.getDouble("learning_rate")
    val tolerance : Double = config.getDouble("tolerance")
    val rateDecay : Double = config.getDouble("rate_decay")
    val score = input.map(x => x._1) // Double
    val label = input.map(x => x._2) // Boolean

    // Transforming the label to target probability
    val n = label.size
    val transformedLabel = label.map(x => if (x) 1.0 else 0.0)

    // Run Batch Gradient Descent
    // y = sigmoid(a+bx), a is the offset, b is the scale
    // initialization
    var a = 0.0
    var b = 1.0
    var old_a = 0.0
    var old_b = 1.0
    var iter = 0

    do {
      log.info("Iteration %d, a = %f; b = %f".format(iter, a, b))
      old_a = a
      old_b = b

      // predicted probability
      val predProb = score.map(x => sigmoid(a + b * x))
      //An array of predProb(j) - transformedLabel(j)
      val predDiff = predProb.zip(transformedLabel).map(x => (x._1 - x._2).toFloat)

      val gradientA = predDiff.sum / n
      val gradientB = predDiff.zip(score).map(x => x._1 * x._2).sum / n

      a -= learningRate * gradientA
      b -= learningRate * gradientB
      learningRate = learningRate * rateDecay
      iter += 1
      log.info("Iteration %d, a = %f; b = %f".format(iter, a, b))
    } while((math.abs(old_a - a) > tolerance ||
             math.abs(old_b - b) > tolerance) && iter < maxIter)
    Array(a, b)
  }
}
