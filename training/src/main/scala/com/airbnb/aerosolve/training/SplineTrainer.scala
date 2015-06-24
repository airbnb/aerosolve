package com.airbnb.aerosolve.training

import java.io.{BufferedWriter, OutputStreamWriter}
import java.util.concurrent.ConcurrentHashMap

import com.airbnb.aerosolve.training.CyclicCoordinateDescent.Params
import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.core.models.{AbstractModel, SplineModel}
import com.airbnb.aerosolve.core.models.SplineModel.WeightSpline
import com.airbnb.aerosolve.core.{Example, FeatureVector}
import com.typesafe.config.Config
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.mutable.HashMap
import scala.util.Try

object SplineTrainer {
  private final val log: Logger = LoggerFactory.getLogger("SplineTrainer")

  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String) : SplineModel = {
    val loss : String = config.getString(key + ".loss")
    val numBins : Int = config.getInt(key + ".num_bins")
    val numBags : Int = config.getInt(key + ".num_bags")
    val iterations : Int = config.getInt(key + ".iterations")
    val rankKey : String = config.getString(key + ".rank_key")
    val learningRate : Double = config.getDouble(key + ".learning_rate")
    val dropout : Double = config.getDouble(key + ".dropout")
    val minCount : Int = config.getInt(key + ".min_count")
    val subsample : Double = config.getDouble(key + ".subsample")
    val linfinityCap : Double = config.getDouble(key + ".linfinity_cap")
    val smoothingTolerance : Double = config.getDouble(key + ".smoothing_tolerance")
    val linfinityThreshold : Double = config.getDouble(key + ".linfinity_threshold")

    val margin : Double = Try(config.getDouble(key + ".margin")).getOrElse(1.0)
    
    val multiscale : Array[Int] = Try(
        config.getIntList(key + ".multiscale").asScala.map(x => x.toInt).toArray)
      .getOrElse(Array[Int]())

    val pointwise : RDD[Example] =
      LinearRankerUtils
        .makePointwiseFloat(input, config, key)

    var model = new SplineModel()
    model.initForTraining(numBins)
    model.setSplineNormCap(linfinityCap.toFloat)
    initModel(minCount, subsample, rankKey, pointwise, model)
    setPrior(config, key, model)
    log.info("Computing min/max values for all features")

    log.info("Training using " + loss)
    for (i <- 1 to iterations) {
      model = if (multiscale.size == 0) {
        sgdTrain(sc,     
               config,
               key,
               pointwise,
               numBins,
               numBags,
               rankKey,
               loss,
               learningRate,
               dropout,
               subsample,
               i,
               margin,
               smoothingTolerance,
               linfinityThreshold,
               model)
      } else {
        sgdMultiscaleTrain(sc,     
               config,
               key,
               pointwise,
               numBins,
               numBags,
               rankKey,
               multiscale,
               loss,
               learningRate,
               dropout,
               subsample,
               i,
               margin,
               smoothingTolerance,
               linfinityThreshold,
               model)
      }
    }
    model
  }

  // Intializes the model
  def initModel(minCount : Int,
                subsample : Double,
                rankKey : String,
                input : RDD[Example],
                model : SplineModel) = {
    val minMax = getMinMax(minCount, rankKey, input.sample(false, subsample))
    log.info("Num features = %d".format(minMax.length))
    for (entry <- minMax) {
      model.addSpline(entry._1._1, entry._1._2, entry._2._1.toFloat, entry._2._2.toFloat)
    }
  }

  def setPrior(config : Config,
               key : String,
               model : SplineModel) = {
    try {
      val priors = config.getStringList(key + ".prior")
      for (prior <- priors) {
        val tokens : Array[String] = prior.split(",")
        if (tokens.length == 4) {
          val family = tokens(0)
          val name = tokens(1)
          val start = tokens(2).toDouble
          val end = tokens(3).toDouble
          val familyMap = model.getWeightSpline.asScala.get(family)
          if (familyMap != None) {
            val spline = familyMap.get.get(name)
            if (spline != null) {
              log.info("Setting prior %s:%s <- %f to %f".format(family, name, start, end))
              val len = spline.splineWeights.length
              for (i <- 0 until len) {
                val t = i.toDouble / (len.toDouble - 1.0)
                spline.splineWeights(i) = ((1.0 - t) * start + t * end).toFloat
              }
            }
          }
        } else {
          log.error("Incorrect number of parameters for %s".format(prior))
        }
      }
    } catch {
      case _ : Throwable => log.info("No prior given")
    }
  }

  // Returns the min/max of a feature
  def getMinMax(minCount : Int,
                  rankKey : String,
                  input : RDD[Example]) : Array[((String, String), (Double, Double))] = {
    input
      .mapPartitions(partition => {
      // family, feature name => min, max, count
      val weights = new ConcurrentHashMap[(String, String), (Double, Double, Int)]().asScala
      partition.foreach(example => {
        val flatFeature = Util.flattenFeature(example.example.get(0)).asScala
        flatFeature.foreach(familyMap => {
          if (!rankKey.equals(familyMap._1)) {
            familyMap._2.foreach(feature => {
              val key = (familyMap._1, feature._1)
              val curr = weights.getOrElse(key,
                                           (Double.MaxValue, -Double.MaxValue, 0))
              weights.put(key,
                          (scala.math.min(curr._1, feature._2),
                           scala.math.max(curr._2, feature._2),
                           curr._3 + 1)
              )
            })
          }
        })
      })
      weights.iterator
    })
    .reduceByKey((a, b) =>
                   (scala.math.min(a._1, b._1),
                    scala.math.max(a._2, b._2),
                    a._3 + b._3))
    .filter(x => x._2._3 >= minCount)
    .map(x => (x._1, (x._2._1, x._2._2)))
    .collect
    .toArray
  }

  def evaluatePolynomial(coeff : Array[Double],
                         data : Array[Double],
                         overwrite : Boolean) : Double = {
    val len = data.length
    var err : Double = 0.0
    var count : Double = 0.0
    for (i <- 0 until len) {
      val t : Double = i.toDouble / (len - 1.0)
      val tinv = 1.0 - t
      val diracStart = if (i == 0) coeff(0) else 0.0
      val diracEnd = if (i == len - 1) coeff(1) else 0.0
      val eval = coeff(2) * tinv * tinv * tinv +
                 coeff(3) * 3.0 * tinv * tinv * t +
                 coeff(4) * 3.0 * tinv * t * t +
                 coeff(5) * t * t * t +
                 diracStart +
                 diracEnd
      if (data(i) != 0.0) {
        err = err + math.abs(eval - data(i))
        count = count + 1.0
      }
      if (overwrite) {
        data(i) = eval
      }
    }
    err / count
  }

  // Fits a polynomial to the data.
  def fitPolynomial(data : Array[Double]) : (Double, Array[Double]) = {
    val numCoeff = 6
    val iterations = numCoeff * 4
    val len = data.length - 1
    val initial = Array.fill(numCoeff)(0.0)
    val initialStep = Array.fill(numCoeff)(1.0)
    val bounds = Array.fill(numCoeff)((-10.0, 10.0))
    val params = Params(1.0 / 512.0, iterations, initial, initialStep, bounds)
    def f(x : Array[Double]) = {
      evaluatePolynomial(x, data, false)
    }
    val best = CyclicCoordinateDescent.optimize(f, params)
    return (f(best), best)
  }

  // Returns true if we manage to fit a polynomial
  def smoothSpline(tolerance : Double,
                   spline : WeightSpline) = {
    val weights = spline.splineWeights
    val optimize = weights.map(x => x.toDouble).toArray
    val errAndCoeff = fitPolynomial(optimize)
    if (errAndCoeff._1 < tolerance) {
      evaluatePolynomial(errAndCoeff._2, optimize, true)
      for (i <- 0 until weights.length) {
        weights(i) = optimize(i).toFloat
      }
    }
  }

  def sgdTrain(sc : SparkContext,
               config : Config,
               key : String,
               input : RDD[Example],
               numBins : Int,
               numBags : Int,
               rankKey : String,
               loss : String,
               learningRate : Double,
               dropout : Double,
               subsample : Double,
               iteration : Int,
               margin : Double,
               smoothingTolerance : Double,
               linfinityThreshold : Double,
               model : SplineModel) : SplineModel = {
    log.info("Iteration %d".format(iteration))

    val modelBC = sc.broadcast(model)

    val threshold : Double = config.getDouble(key + ".rank_threshold")


    val lossMod : Int = try {
      config.getInt(key + ".loss_mod")
    } catch {
      case _ : Throwable => 100
    }

    input
      .sample(false, subsample)
      .coalesce(numBags, true)
      .mapPartitions(partition =>
        sgdPartition(partition,
          modelBC, loss, lossMod, threshold, learningRate, dropout, margin,
          rankKey))
    .groupByKey
    // Average the spline weights
    .map(x => {
      val head = x._2.head
      val spline = new WeightSpline(head.spline.getMinVal,
                                    head.spline.getMaxVal,
                                    numBins)
      val scale = 1.0f / numBags.toFloat
      x._2.foreach(entry => {
        for (i <- 0 until numBins) {
          spline.splineWeights(i) = spline.splineWeights(i) + scale * entry.splineWeights(i)
        }
      })
      smoothSpline(smoothingTolerance, spline)
      (x._1, spline)
    })
    .collect
    .foreach(entry => {
      val family = model.getWeightSpline.get(entry._1._1)
      if (family != null && family.containsKey(entry._1._2)) {
        family.put(entry._1._2, entry._2)
      }
    })

    deleteSmallSplines(model, linfinityThreshold)

    TrainingUtils.saveModel(model, config, key + ".model_output")
    return model
  }
  
  def sgdMultiscaleTrain(sc : SparkContext,
                         config : Config,
                         key : String,
                         input : RDD[Example],
                         numBins : Int,
                         numBags : Int,
                         rankKey : String,
                         multiscale : Array[Int],
                         loss : String,
                         learningRate : Double,
                         dropout : Double,
                         subsample : Double,
                         iteration : Int,
                         margin : Double,
                         smoothingTolerance : Double,
                         linfinityThreshold : Double,
                         model : SplineModel) : SplineModel = {
    log.info("Multiscale Iteration %d".format(iteration))

    val modelBC = sc.broadcast(model)

    val threshold : Double = config.getDouble(key + ".rank_threshold")

    val lossMod : Int = try {
      config.getInt(key + ".loss_mod")
    } catch {
      case _ : Throwable => 100
    }

    input
      .sample(false, subsample)
      .coalesce(numBags, true)
      .mapPartitionsWithIndex((index, partition) =>
        sgdPartitionMultiscale(index, partition, multiscale,
          modelBC, loss, lossMod, threshold, learningRate, dropout, margin,
          rankKey))
    .groupByKey
    // Average the spline weights
    .map(x => {
      val head = x._2.head
      val spline = new WeightSpline(head.spline.getMinVal,
                                    head.spline.getMaxVal,
                                    numBins)
      val scale = 1.0f / numBags.toFloat
      x._2.foreach(entry => {
        entry.resample(numBins)
        for (i <- 0 until numBins) {
          spline.splineWeights(i) = spline.splineWeights(i) + scale * entry.splineWeights(i)
        }
      })
      smoothSpline(smoothingTolerance, spline)
      (x._1, spline)
    })
    .collect
    .foreach(entry => {
      val family = model.getWeightSpline.get(entry._1._1)
      if (family != null && family.containsKey(entry._1._2)) {
        family.put(entry._1._2, entry._2)
      }
    })

    deleteSmallSplines(model, linfinityThreshold)

    TrainingUtils.saveModel(model, config, key + ".model_output")
    return model
  }

  
  def deleteSmallSplines(model : SplineModel,
                         linfinityThreshold : Double) = {
    val toDelete = scala.collection.mutable.ArrayBuffer[(String, String)]()

    model.getWeightSpline.asScala.foreach(family => {
      family._2.asScala.foreach(entry => {
        if (entry._2.LInfinityNorm < linfinityThreshold) {
          toDelete.append((family._1, entry._1))
        }
      })
    })

    log.info("Deleting %d empty splines".format(toDelete.size))

    toDelete.foreach(entry => {
      val family = model.getWeightSpline.get(entry._1)
      if (family != null && family.containsKey(entry._2)) {
        family.remove(entry._2)
      }
    })
  }
  
  def sgdPartition(partition : Iterator[Example],
                   modelBC : Broadcast[SplineModel],
                   loss : String,
                   lossMod : Int,
                   threshold : Double,
                   learningRate : Double,
                   dropout : Double,
                   margin : Double,
                   rankKey : String) = {
    val workingModel = modelBC.value
    val output = sgdPartitionInternal(partition, workingModel, loss, lossMod, threshold, learningRate, dropout, margin, rankKey)
    output.iterator
  }
  
  def sgdPartitionMultiscale(
       index : Int,
       partition : Iterator[Example],
       multiscale : Array[Int],
       modelBC : Broadcast[SplineModel],
       loss : String,
       lossMod : Int,
       threshold : Double,
       learningRate : Double,
       dropout : Double,
       margin : Double,
       rankKey : String) = {
    val workingModel = modelBC.value
    
    val newBins = multiscale(index % multiscale.size)
    
    log.info("Resampling to %d bins".format(newBins))
    workingModel
      .getWeightSpline
      .foreach(family => {
        family._2.foreach(feature => {
          feature._2.resample(newBins)
        })
    })
    
    val output = sgdPartitionInternal(partition, workingModel, loss, lossMod, threshold, learningRate, dropout, margin, rankKey)
    output.iterator
  }
  
  def sgdPartitionInternal(partition : Iterator[Example],
                           workingModel : SplineModel,
                           loss : String,
                           lossMod : Int,
                           threshold : Double,
                           learningRate : Double,
                           dropout : Double,
                           margin : Double,
                           rankKey : String) : HashMap[(String, String), SplineModel.WeightSpline] = {
    @volatile var lossSum : Double = 0.0
    @volatile var lossCount : Int = 0
    partition.foreach(example => {
      val fv = example.example.get(0)
      val rank = fv.floatFeatures.get(rankKey).asScala.head._2
      val label = if (rank <= threshold) {
        -1.0
      } else {
        1.0
      }
      loss match {
        case "logistic" => lossSum = lossSum + updateLogistic(workingModel, fv, label, learningRate,dropout)
        case "hinge" => lossSum = lossSum + updateHinge(workingModel, fv, label, learningRate, dropout, margin)
        case _ => {
          log.error("Unknown loss function %s".format(loss))
          System.exit(-1)
        }
      }
      lossCount = lossCount + 1
      if (lossCount % lossMod == 0) {
        log.info("Loss = %f, samples = %d".format(lossSum / lossMod.toDouble, lossCount))
        lossSum = 0.0
      }
    })
    val output = HashMap[(String, String), SplineModel.WeightSpline]()
    workingModel
      .getWeightSpline
      .foreach(family => {
        family._2.foreach(feature => {
          output.put((family._1, feature._1), feature._2)
        })
    })
    output
  }
  
  // http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
  // We rescale by 1 / p so that at inference time we don't have to scale by p.
  // In our case p = 1.0 - dropout rate
  def updateLogistic(model : SplineModel,
                     fv : FeatureVector,
                     label : Double,
                     learningRate : Double,
                     dropout : Double) : Double = {
    val flatFeatures = Util.flattenFeatureWithDropout(fv, dropout)
    val prediction = model.scoreFlatFeatures(flatFeatures) / (1.0 - dropout)
    // To prevent blowup.
    val corr = scala.math.min(10.0, label * prediction)
    val expCorr = scala.math.exp(corr)
    val loss = scala.math.log(1.0 + 1.0 / expCorr)
    val grad = -label / (1.0 + expCorr)
    model.update(grad.toFloat,
                 learningRate.toFloat,
                 flatFeatures)
    return loss
  }

  def updateHinge(model : SplineModel,
                  fv : FeatureVector,
                  label : Double,
                  learningRate : Double,
                  dropout : Double,
                  margin : Double) : Double = {
    val flatFeatures = Util.flattenFeatureWithDropout(fv, dropout)
    val prediction = model.scoreFlatFeatures(flatFeatures) / (1.0 - dropout)
    val loss = scala.math.max(0.0, margin - label * prediction)
    if (loss > 0.0) {
      val grad = -label
      model.update(grad.toFloat,
                   learningRate.toFloat,
                   flatFeatures)
    }
    return loss
  }

  def trainAndSaveToFile(sc : SparkContext,
                         input : RDD[Example],
                         config : Config,
                         key : String) = {
    val model = train(sc, input, config, key)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }
}
