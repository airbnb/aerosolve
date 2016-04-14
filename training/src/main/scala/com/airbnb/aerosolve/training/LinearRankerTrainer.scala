package com.airbnb.aerosolve.training

import java.io.BufferedWriter
import java.io.OutputStreamWriter
import java.util.concurrent.ConcurrentHashMap

import com.airbnb.aerosolve.core.features._
import com.airbnb.aerosolve.core.{ModelRecord, ModelHeader, FeatureVector, Example}
import com.airbnb.aerosolve.core.models.LinearModel
import com.airbnb.aerosolve.core.util.Util
import com.typesafe.config.Config
import org.apache.spark.broadcast.Broadcast
import org.slf4j.{LoggerFactory, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.util.Try
import scala.util.Random
import scala.math.abs
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import scala.collection.mutable

/*
 * A trainer that generates a linear ranker.
 *
 * Params:
 * loss - "ranking" (sgd), "regression" (sgd)
 * learning_rate - a number between 0 and 1
 * num_bags - number of parallel models to make. sgd only.
 *
 * For ranking:
 *
 * The optimization objective is score(positive) > score(negative) + 1
 * We model this as loss = 1 - pos_score + neg_score
 * but since score(x) = w' . x
 * loss = 1 - w'(pos - neg)
 * and dloss/dw = -pos + neg
 * Since we are doing gradient descent, -dloss/dw = pos - neg
 * For binary features this is the set of positive and negative features
 * that are different.
 * For the gradient update step we use adagrad which dynamically adjusts
 * the step size based on the magnitude of the previous gradients.
 * http://www.magicbroom.info/Papers/DuchiHaSi10.pdf
 *
 * For regression:
 * epsilon - a parameter such that if |target - prediction| < eps
 *   no update will be taken
 *
 * Otherwise the gradient is the epsilon insensitive loss
 * i.e. if w * x - target > epsilon, the gradient step is -1, else +1
 *
 * For classification
 * rank_threshold - anything smaller or equal to this is a negative.
 *
 * if w * x * label > 1 we are correct. Else take a step - sign(w * x)
 *
 */

object LinearRankerTrainer {
  private final val log: Logger = LoggerFactory.getLogger("LinearRankerTrainer")
  private final val lossKey = ("$loss", "$loss")
  val MAX_WEIGHTS : Int = 1000000

  def pickTrainer(sc : SparkContext,
                  input : RDD[Example],
                  config : Config,
                  key : String,
                  loss : String,
                  numBags : Int,
                  weights : collection.mutable.Map[Feature, (Double, Double)],
                  iteration : Int) :
  RDD[(Feature, (Double, Double))] = {
    val registry : FeatureRegistry = new FeatureRegistry
    val labelFamily: Family = registry.family(config.getString(key + ".rank_key"))
    val lossFeature : Feature = registry.feature(lossKey._1, lossKey._2)
    val lambda = config.getDouble(key + ".lambda")
    val lambda2 = config.getDouble(key + ".lambda2")
    val learningRate = config.getDouble(key + ".learning_rate")

    val dropout = if (config.hasPath(key + ".dropout")) {
      Option(config.getDouble(key + ".dropout"))
    } else None
    val weightsBC = sc.broadcast(weights)
    val examples = LinearRankerUtils
      .makePointwiseFloat(input, config, key, registry)
      .coalesce(numBags, true)

    // The following code might be a bit confusing for people who aren't experienced with Scala.
    // To re-use more code, I'm using currying and function passing.
    // If you're more used to OOP, imagine each type of loss is a subclass with a bunch of
    // constructor params. Each class implements .updateWeights for a vector.
    // Ranking is different and (conceptually) overrides the default .updateWeights for an example
    // so it can behave a little differently.
    // Since OOP can be implemented with closures and currying is simplified closures, these are
    // just two ways to do the same thing.
    // I think this functional approach is actually a lot easier to write and read even though
    // it's likely a little unfamiliar.
    // TODO (Brad) : This can all be simplified a lot more.
    val weightUpdateFunction = if (loss == "ranking") {
      ranking(lambda, lambda2, learningRate, lossFeature) _
    } else {
      val function = loss match {
        case "regression" => regression(lambda, lambda2, config.getDouble(key + ".epsilon"),
                                        learningRate, lossFeature) _
        case "regressionL2" => regressionL2(lambda, lambda2, learningRate, lossFeature) _
        case "hinge" => classification(lambda, lambda2, learningRate,
                                       config.getDouble(key + ".rank_threshold"), lossFeature) _
        case "logistic" => logistic(lambda, lambda2, learningRate,
                                    config.getDouble(key + ".rank_threshold"), lossFeature) _
        case _ => {
          val message = "Unknown loss type %s".format(loss)
          log.error(message)
          throw new IllegalArgumentException(message)
        }
      }
      basicWeightUpdateFunction(function) _
    }

    updateWeights(examples, weightsBC, iteration, labelFamily, dropout, weightUpdateFunction)
  }

  def updateWeights(input : RDD[Example],
            weightsBC : Broadcast[collection.mutable.Map[Feature, (Double, Double)]],
            iteration : Int,
            labelFamily : Family,
            dropout : Option[Double],
            weightUpdateFunction : (Iterable[MultiFamilyVector],
              collection.mutable.Map[Feature, (Double, Double)],
              Family, Option[Double], Int) => Unit): RDD[(Feature, (Double, Double))] = {
    input
      .mapPartitions(partition => {
        val weightMap = weightsBC.value
        partition.foreach(examples => {
          val vectors = examples
            .filter(vector => vector.contains(labelFamily))
          weightUpdateFunction(vectors, weightMap, labelFamily, dropout, iteration)
        })
        weightMap.iterator
      })
  }

  def basicWeightUpdateFunction(function:
                                 (Iterable[Feature],
                                  collection.mutable.Map[Feature, (Double, Double)],
                                  Double, Double, Int) => Unit)
                                 (vectors: Iterable[MultiFamilyVector],
                                  weightMap: collection.mutable.Map[Feature, (Double, Double)],
                                  labelFamily: Family,
                                  dropout: Option[Double],
                                  iteration: Int): Unit = {
    vectors.foreach(vector => {
      val label = LinearRankerUtils.getLabel(vector, labelFamily)
      val scoreVector: Iterable[FeatureValue] = dropout
        .map(d => vector.withDropout(d).asScala)
        .getOrElse(vector.iterator.asScala.toIterable)
      val features = LinearRankerUtils.getFeatures(scoreVector)
      val prediction = LinearRankerUtils.score(features, weightMap)
      val finalPrediction = dropout
        .map(dropoutVal => prediction / (1.0 - dropoutVal))
        .getOrElse(prediction)
      function(features, weightMap, finalPrediction, label, iteration)
    })
  }

  def ranking(lambda: Double,
              lambda2: Double,
              learningRate: Double,
              lossFeature : Feature)
             (vectors: Iterable[MultiFamilyVector],
              weightMap: collection.mutable.Map[Feature, (Double, Double)],
              labelFamily: Family,
              dropout: Option[Double],
              iteration: Int): Unit = {
    val rnd = new Random(java.util.Calendar.getInstance().getTimeInMillis)
    var size = weightMap.size
    LinearRankerUtils.rankingCompression(vectors, labelFamily)
      .foreach(ce => {
        val pos = ce.pos.filter(x => rnd.nextDouble() > dropout.get)
        val neg = ce.neg.filter(x => rnd.nextDouble() > dropout.get)
        val posScore = LinearRankerUtils.score(pos, weightMap) / (1.0 - dropout.get)
        val negScore = LinearRankerUtils.score(neg, weightMap) / (1.0 - dropout.get)
        val loss = 1.0 - posScore + negScore
        val lossEntry = weightMap.getOrElse(lossFeature, (0.0, 0.0))
        if (loss > 0.0) {
          def update(feature : Feature, grad : Double) = {
            val wt = weightMap.getOrElse(feature, (0.0, 0.0))
            // Allow puts only in the first iteration and size is less than MAX_SIZE
            // or if it already exists.
            if ((iteration == 1 && size < MAX_WEIGHTS) || wt._1 != 0.0) {
              val newGradSum = wt._2 + 1.0
              val newWeight = fobosUpdate(currWeight = wt._1,
                                          gradient = grad,
                                          eta =  learningRate,
                                          l1Reg = lambda,
                                          l2Reg = lambda2,
                                          sum = newGradSum)
              if (newWeight == 0.0) {
                weightMap.remove(feature)
              } else {
                weightMap.put(feature, (newWeight, newGradSum))
              }
              if (wt._1 == 0.0) {
                size = size + 1
              }
            }
          }
          pos.foreach(v => {
            update(v, -1.0)
          })
          neg.foreach(v => {
            update(v, 1.0)
          })
          weightMap.put(lossFeature, (lossEntry._1 + loss, lossEntry._2 + 1.0))
        } else {
          weightMap.put(lossFeature, (lossEntry._1, lossEntry._2 + 1.0))
        }
      })
  }
  
  def regression(lambda: Double,
                     lambda2: Double,
                     epsilon: Double,
                     learningRate: Double,
                     lossFeature : Feature)
                    (features : Iterable[Feature],
                     weightMap: collection.mutable.Map[Feature, (Double, Double)],
                     prediction: Double,
                     target : Double,
                     iteration: Int) : Unit = {
    // The loss function the epsilon insensitive loss L = max(0,|w'x - y| - epsilon)
    // So if prediction = w'x and prediction > y then
    // the dloss / dw is
    
    val diff = prediction - target
    val loss = Math.abs(diff) - epsilon
    val lossEntry = weightMap.getOrElse(lossFeature, (0.0, 0.0))

    if (loss <= 0) {
      // No loss suffered
      weightMap.put(lossFeature, (lossEntry._1, lossEntry._2 + 1.0))
    } else {
      val grad = if (diff > 0) 1.0 else -1.0
      features.foreach(feature => {
        val wt = weightMap.getOrElse(feature, (0.0, 0.0))
        val newGradSum = wt._2 + 1.0
        val newWeight = fobosUpdate(currWeight = wt._1,
                                    gradient = grad,
                                    eta = learningRate,
                                    l1Reg = lambda,
                                    l2Reg = lambda2,
                                    sum = newGradSum)
        weightMap.put(feature, (newWeight, newGradSum))
      })
      weightMap.put(lossFeature, (lossEntry._1 + loss, lossEntry._2 + 1.0))
    }
  }

  def regressionL2(lambda: Double,
                       lambda2: Double,
                       learningRate: Double,
                       lossFeature : Feature)
                      (features : Iterable[Feature],
                       weightMap: collection.mutable.Map[Feature, (Double, Double)],
                       prediction: Double,
                       target : Double,
                       iteration : Int) : Unit = {
    // The loss function is the squared error loss L = 0.5 * ||w'x - y||^2
    // So dloss / dw = (w'x - y) * x

    val diff = prediction - target
    val sqdiff = diff * diff
    val loss = 0.5 * sqdiff
    val lossEntry = weightMap.getOrElse(lossFeature, (0.0, 0.0))

    val grad = diff
    features.foreach(feature => {
      val wt = weightMap.getOrElse(feature, (0.0, 0.0))
      val newGradSum = wt._2 + sqdiff
      val newWeight = fobosUpdate(currWeight = wt._1,
                                  gradient = grad,
                                  eta = learningRate,
                                  l1Reg = lambda,
                                  l2Reg = lambda2,
                                  sum = newGradSum)
      weightMap.put(feature, (newWeight, newGradSum))
    })
    weightMap.put(lossFeature, (lossEntry._1 + loss, lossEntry._2 + 1.0))
  }

  def classification(lambda : Double,
                         lambda2 : Double,
                         learningRate : Double,
                         threshold : Double,
                         lossFeature : Feature)
                        (features : Iterable[Feature],
                         weightMap: collection.mutable.Map[Feature, (Double, Double)],
                         prediction: Double,
                         rank : Double,
                         iteration : Int) : Unit = {
    val label = if (rank <= threshold) {
      -1.0
    } else {
      1.0
    }
    val loss = 1.0 - label * prediction
    val lossEntry = weightMap.getOrElse(lossFeature, (0.0, 0.0))
    var size = weightMap.size
    if (loss > 0.0) {
      features.foreach(feature => {
        val wt = weightMap.getOrElse(feature, (0.0, 0.0))
        // Allow puts only in the first iteration and size is less than MAX_SIZE
        // or if it already exists.
        if ((iteration == 1 && size < MAX_WEIGHTS) || wt._1 != 0.0) {
          if (wt._1 == 0.0) {
            // We added a weight increase the size.
            size = size + 1
          }
          val newGradSum = wt._2 + 1.0
          val newWeight = fobosUpdate(currWeight = wt._1,
                                      gradient = -label,
                                      eta = learningRate,
                                      l1Reg = lambda,
                                      l2Reg = lambda2,
                                      sum = newGradSum)
          if (newWeight == 0.0) {
            weightMap.remove(feature)
          } else {
            weightMap.put(feature, (newWeight, newGradSum))
          }
        }
      })
      weightMap.put(lossFeature, (lossEntry._1 + loss, lossEntry._2 + 1.0))
    } else {
      weightMap.put(lossFeature, (lossEntry._1, lossEntry._2 + 1.0))
    }
  }

  def logistic(lambda : Double,
               lambda2 : Double,
               learningRate : Double,
               threshold : Double,
               lossFeature : Feature)
              (features : Iterable[Feature],
               weightMap: collection.mutable.Map[Feature, (Double, Double)],
               prediction: Double,
               rank : Double,
               iteration : Int) : Unit = {
    val label = if (rank <= threshold) {
      -1.0
    } else {
      1.0
    }
    // To prevent blowup.
    val corr = scala.math.min(10.0, label * prediction)
    val expCorr = scala.math.exp(corr)
    val loss = scala.math.log(1.0 + 1.0 / expCorr)
    val lossEntry = weightMap.getOrElse(lossFeature, (0.0, 0.0))
    var size = weightMap.size
    features.foreach(feature => {
      val wt = weightMap.getOrElse(feature, (0.0, 0.0))
      // Allow puts only in the first iteration and size is less than MAX_SIZE
      // or if it already exists.
      if ((iteration == 1 && size < MAX_WEIGHTS) || wt._1 != 0.0) {
        if (wt._1 == 0.0) {
          // We added a weight increase the size.
          size = size + 1
        }
        val newGradSum = wt._2 + 1.0
        val grad = -label / (1.0 + expCorr)
        val newWeight = fobosUpdate(currWeight = wt._1,
                                    gradient = grad,
                                    eta = learningRate,
                                    l1Reg = lambda,
                                    l2Reg = lambda2,
                                    sum = newGradSum)
        if (newWeight == 0.0) {
          weightMap.remove(feature)
        } else {
          weightMap.put(feature, (newWeight, newGradSum))
        }
      }
    })
  }


  // http://papers.nips.cc/paper/3793-efficient-learning-using-forward-backward-splitting.pdf
  def fobosUpdate(currWeight : Double,
                  gradient : Double,
                  eta : Double,
                  l1Reg : Double,
                  l2Reg : Double,
                  sum : Double) : Double = {
    val etaT = eta / scala.math.sqrt(sum)
    val etaTHalf = eta / scala.math.sqrt(sum + 0.5)
    // FOBOS l2 regularization
    val wt = (currWeight - gradient * etaT) / (1.0 + l2Reg * etaTHalf)
    // FOBOS l1 regularization
    val sign = if (wt > 0.0) 1.0 else -1.0
    val step = scala.math.max(0.0, scala.math.abs(wt) - l1Reg * etaTHalf)
    sign * step
  }

  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String,
             registry : FeatureRegistry) : Array[(Feature, Double)] = {
    val loss: String = try {
      config.getString(key + ".loss")
    } catch {
      case _: Throwable => "ranking"
    }
    log.info("Training using " + loss)
    sgdTrain(sc, input, config, key, loss, registry)
  }

  def setPrior(config : Config,
               key : String,
               weights : mutable.Map[Feature, (Double, Double)],
                registry : FeatureRegistry) = {
    try {
      val priors = config.getStringList(key + ".prior")
      for (prior <- priors) {
        val tokens : Array[String] = prior.split(",")
        if (tokens.length == 3) {
          val family = tokens(0)
          val name = tokens(1)
          val weight = tokens(2).toDouble
          log.info("Setting prior %s:%s = %f".format(family, name, weight))
          weights.put(registry.feature(family, name), (weight, 1.0))
        }
      }
    } catch {
      case _ : Throwable => log.info("No prior given")
    }
  }

  def sgdTrain(sc : SparkContext,
               input : RDD[Example],
               config : Config,
               key : String,
               loss : String,
               registry : FeatureRegistry) : Array[(Feature, Double)] = {
    val numBags : Int = config.getInt(key + ".num_bags")
    val iterations : Int = config.getInt(key + ".iterations")
    val subsample : Double = Try(config.getDouble(key + ".subsample")).getOrElse(1.0)

    // The keys the feature (family, value)
    // The values are the weight.
    var weights = mutable.HashMap[Feature, (Double, Double)]()
    setPrior(config, key, weights, registry)
    // Since we are bagging models, average them by numBags
    val scale : Double = 1.0 / numBags.toDouble

    for (i <- 1 to iterations) {
      log.info("Iteration %d".format(i))
      val filteredInput = input
        .filter(examples => examples != null)
        .sample(false, subsample)
      val resultsRDD = pickTrainer(sc, filteredInput, config, key, loss, numBags, weights, i)
        .reduceByKey((a,b) => (a._1 + b._1, a._2 + b._2))
        .persist()

      val lossFeature = registry.feature(lossKey._1, lossKey._2)
      val lossV = resultsRDD.filter(x => x._1 == lossFeature).take(1)
      var lossSum = 0.0
      var count = 0.0
      if (!lossV.isEmpty) {
        lossSum = lossV.head._2._1
        count = lossV.head._2._2
      } else {
        0.0
      }
      val results = resultsRDD
        .filter(x => x._1 != lossFeature)
        .map(x => (scala.math.abs(x._2._1), (x._1, x._2)))
        .top(LinearRankerTrainer.MAX_WEIGHTS)
        .map(x => x._2)

      // Nuke the old weights
      weights = mutable.HashMap[Feature, (Double, Double)]()
      var sz = 0
      results
        .foreach(value => {
        sz = sz + 1
        weights.put(value._1, (value._2._1 * scale, value._2._2))
      })
      log.info("Average loss = %f count = %f weight size = %d".format(lossSum / count, count, sz))
      resultsRDD.unpersist()
    }
    weights
      // Strip off the sum of squared gradients
      .map(x => (x._1, x._2._1))
      .toBuffer
      .sortWith((x, y) => abs(x._2) > abs(y._2))
      .toArray
  }

  def save(writer : BufferedWriter, weights : Array[(Feature, Double)]) = {
    val header = new ModelHeader()
    header.setModelType("linear")
    header.setNumRecords(weights.size)
    val headerRecord = new ModelRecord()
    headerRecord.setModelHeader(header)
    writer.write(Util.encode(headerRecord))
    writer.write('\n')
    log.info("Top 50 weights")
    for(i <- 0 until weights.size) {
      val weight = weights(i)
      val family = weight._1.family.name
      val name = weight._1.name
      val wt = weight._2
      if (i < 50) {
        log.info("%s : %s = %f".format(family, name, wt))
      }
      val record = new ModelRecord()
      record.setFeatureFamily(family)
      record.setFeatureName(name)
      record.setFeatureWeight(wt)
      writer.write(Util.encode(record))
      writer.write('\n')
    }
    writer.close
  }

  def trainAndSaveToFile(sc : SparkContext,
                         input : RDD[Example],
                         config : Config,
                         key : String,
                          registry: FeatureRegistry) = {
    val weights = train(sc, input, config, key, registry)
    val output : String = config.getString(key + ".model_output")
    val fileSystem = FileSystem.get(new java.net.URI(output),
                                    new Configuration())
    val file = fileSystem.create(new Path(output), true)
    val writer = new BufferedWriter(new OutputStreamWriter(file))
    save(writer, weights)
  }
}
