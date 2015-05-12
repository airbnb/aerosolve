package com.airbnb.aerosolve.training

import java.io.{BufferedWriter, OutputStreamWriter}
import java.util.concurrent.ConcurrentHashMap

import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.core.models.MaxoutModel
import com.airbnb.aerosolve.core.{Example, FeatureVector}
import com.typesafe.config.Config
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._


object MaxoutTrainer {
  private final val log: Logger = LoggerFactory.getLogger("MaxoutTrainer")

  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String) : MaxoutModel = {
    val loss : String = config.getString(key + ".loss")
    val numHidden : Int = config.getInt(key + ".num_hidden")
    val iterations : Int = config.getInt(key + ".iterations")
    val rankKey : String = config.getString(key + ".rank_key")
    val learningRate : Double = config.getDouble(key + ".learning_rate")
    val lambda : Double = config.getDouble(key + ".lambda")
    val lambda2 : Double = config.getDouble(key + ".lambda2")
    val dropout : Double = config.getDouble(key + ".dropout")
    val dropoutHidden : Double = config.getDouble(key + ".dropout_hidden")
    val minCount : Int = config.getInt(key + ".min_count")
    val subsample : Double = config.getDouble(key + ".subsample")
    val momentum : Double = config.getDouble(key + ".momentum")

    val pointwise : RDD[Example] =
      LinearRankerUtils
        .makePointwiseFloat(input, config, key)
        .cache()

    var model = new MaxoutModel()
    model.initForTraining(numHidden)
    initModel(minCount, rankKey, pointwise, model)
    log.info("Computing max values for all features")

    log.info("Training using " + loss)
    for (i <- 1 to iterations) {
      model = sgdTrain(sc,
               config,
               key,
               pointwise,
               numHidden,
               rankKey,
               loss,
               learningRate,
               lambda,
               lambda2,
               dropout,
               dropoutHidden,
               momentum,
               subsample,
               i,
               model)
    }
    pointwise.unpersist()
    model
  }

  // Intializes the model
  def initModel(minCount : Int,
                rankKey : String,
                input : RDD[Example],
                model : MaxoutModel) = {
    val maxScale = getMaxScale(minCount, rankKey, input)
    log.info("Num features = %d".format(maxScale.length))
    for (entry <- maxScale) {
      model.addVector(entry._1._1, entry._1._2, entry._2.toFloat)
    }
  }

  // Returns 1 / largest absolute value of the feature
  def getMaxScale(minCount : Int,
                  rankKey : String,
                  input : RDD[Example]) : Array[((String, String), Double)] = {
    input
      .mapPartitions(partition => {
      val weights = new ConcurrentHashMap[(String, String), (Double, Int)]().asScala
      partition.foreach(example => {
        val flatFeature = Util.flattenFeature(example.example.get(0)).asScala
        flatFeature.foreach(familyMap => {
          if (!rankKey.equals(familyMap._1)) {
            familyMap._2.foreach(feature => {
              val key = (familyMap._1, feature._1)
              val curr = weights.getOrElse(key, (0.0, 0))
              weights.put(key, (scala.math.max(curr._1, feature._2), curr._2 + 1))
            })
          }
        })
      })
      weights.iterator
    })
    .reduceByKey((a, b) => (scala.math.max(a._1, b._1), a._2 + b._2))
    .filter(x => x._2._1 > 1e-10 && x._2._2 >= minCount)
    .map(x => (x._1, 1.0 / x._2._1))
    .collect
    .toArray
  }

  def sgdTrain(sc : SparkContext,
               config : Config,
               key : String,
               input : RDD[Example],
               numHidden : Int,
               rankKey : String,
               loss : String,
               learningRate : Double,
               lambda : Double,
               lambda2 : Double,
               dropout : Double,
               dropoutHidden : Double,
               momentum : Double,
               subsample : Double,
               iteration : Int,
               model : MaxoutModel) : MaxoutModel  = {
    log.info("Iteration %d".format(iteration))

    val modelBC = sc.broadcast(model)

    val threshold : Double = config.getDouble(key + ".rank_threshold")

    val lossMod : Int = try {
      config.getInt(key + ".loss_mod")
    } catch {
      case _ : Throwable => 100
    }

    val modelRet = input
      .sample(false, subsample)
      .coalesce(1, true)
      .mapPartitions(partition => {
      val workingModel = modelBC.value
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
          case "logistic" => lossSum = lossSum + updateLogistic(workingModel, fv, label, learningRate, lambda, lambda2, dropout, dropoutHidden, momentum)
          case "hinge" => lossSum = lossSum + updateHinge(workingModel, fv, label, learningRate, lambda, lambda2, dropout, dropoutHidden, momentum)
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
      Array[MaxoutModel](workingModel).iterator
    })
    .collect
    .head
    saveModel(modelRet, config, key)
    return modelRet
  }

  def updateLogistic(model : MaxoutModel,
                     fv : FeatureVector,
                     label : Double,
                     learningRate : Double,
                     lambda : Double,
                     lambda2 : Double,
                     dropout : Double,
                     dropoutHidden : Double,
                     momentum : Double) : Double = {
    val flatFeatures = Util.flattenFeatureWithDropout(fv, dropout)
    val response = model.getResponse(flatFeatures)
    val values = response.getValues
    for (i <- 0 until values.length) {
      if (scala.util.Random.nextDouble() < dropoutHidden) {
        values(i) = 0
      }
    }
    val result = response.getMinMaxResult
    val prediction = result.maxValue - result.minValue
    // To prevent blowup.
    val corr = scala.math.min(10.0, label * prediction)
    val expCorr = scala.math.exp(corr)
    val loss = scala.math.log(1.0 + 1.0 / expCorr)
    val grad = -label / (1.0 + expCorr)
    model.update(grad.toFloat,
                 learningRate.toFloat,
                 lambda.toFloat,
                 lambda2.toFloat,
                 momentum.toFloat,
                 result,
                 flatFeatures)
    return loss
  }

  def updateHinge(model : MaxoutModel,
                  fv : FeatureVector,
                  label : Double,
                  learningRate : Double,
                  lambda : Double,
                  lambda2 : Double,
                  dropout : Double,
                  dropoutHidden : Double,
                  momentum : Double) : Double = {
    val flatFeatures = Util.flattenFeatureWithDropout(fv, dropout)
    val response = model.getResponse(flatFeatures)
    val values = response.getValues
    for (i <- 0 until values.length) {
      if (scala.util.Random.nextDouble() < dropoutHidden) {
        values(i) = 0
      }
    }
    val result = response.getMinMaxResult
    val prediction = result.maxValue - result.minValue
    val loss = scala.math.max(0.0, 1.0 - label * prediction)
    if (loss > 0.0) {
      val grad = -label
      model.update(grad.toFloat,
                   learningRate.toFloat,
                   lambda.toFloat,
                   lambda2.toFloat,
                   momentum.toFloat,
                   result,
                   flatFeatures)
    }
    return loss
  }

  def saveModel(model : MaxoutModel,
                config : Config,
                key : String) = {
    try {
      val output: String = config.getString(key + ".model_output")
      val fileSystem = FileSystem.get(new java.net.URI(output),
                                      new Configuration())
      val file = fileSystem.create(new Path(output), true)
      val writer = new BufferedWriter(new OutputStreamWriter(file))
      model.save(writer)
      writer.close()
      file.close()
    } catch {
      case _ : Throwable => log.error("Could not save model")
    }
  }

  def trainAndSaveToFile(sc : SparkContext,
                         input : RDD[Example],
                         config : Config,
                         key : String) = {
    val model = train(sc, input, config, key)
    saveModel(model, config, key)
  }
}
