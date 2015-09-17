package com.airbnb.aerosolve.training

import java.io.BufferedWriter
import java.io.InputStreamReader
import java.io.BufferedReader
import java.io.OutputStreamWriter
import java.net.URI
import java.util.concurrent.ConcurrentHashMap

import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.core.FeatureVector
import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.core.models.AbstractModel
import com.airbnb.aerosolve.core.models.ModelFactory
import com.airbnb.aerosolve.core.transforms.Transformer
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.slf4j.{LoggerFactory, Logger}

import scala.collection.JavaConverters._

import com.typesafe.config.Config

import scala.collection.mutable.ArrayBuffer

object TrainingUtils {

  val log: Logger = LoggerFactory.getLogger("TrainingUtils")
  val hadoopConfiguration = new Configuration()

  def existsDir(dir: String) = {
    val path = new Path(dir)
    val hdfs = FileSystem.get(
      new java.net.URI(dir), hadoopConfiguration)

    hdfs.exists(path)
  }

  def existsPath(filePath: String) = {
    val path = new Path(filePath)
    val hdfs = FileSystem.get(new java.net.URI(filePath), hadoopConfiguration)

    hdfs.globStatus(path).length > 0
  }

  def deleteDir(dir: String) = {
    val path = new Path(dir)
    val hdfs = FileSystem.get(
      new java.net.URI(dir), hadoopConfiguration)
    if (hdfs.exists(path)) {
      log.warn(dir + " exists, DELETING")
      try {
        hdfs.delete(path, true)
      } catch {
        case _: Throwable => Unit
      }
    }
  }

  def saveModel(model : AbstractModel,
                config : Config,
                key : String) = {
    try {
      val output = config.getString(key)
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

  def loadScoreModel(modelName: String): Option[AbstractModel] = {
    val fs = FileSystem.get(new URI(modelName), hadoopConfiguration)
    val modelPath = new Path(modelName)
    if (!fs.exists(modelPath)) {
      log.error(modelName + " does not exist")
      System.exit(-1)
    }
    val modelStream = fs.open(modelPath)
    val reader = new BufferedReader(new InputStreamReader(modelStream))
    val modelOpt = ModelFactory.createFromReader(reader)
    if (!modelOpt.isPresent) {
      return None
    }
    val model = modelOpt.get()
    return Some(model)
  }

  def loadPlattScaleWeights(filePath: String): Option[Array[Double]] = {
    val calibrationModelStr =
      TrainingUtils.readStringFromFile(filePath)
    if (calibrationModelStr == null) {
      // error already logged in readStringFromFile
      return None
    }

    val calibrationWeights = scala.collection.mutable.ArrayBuffer.empty[Double]
    for (weight <- calibrationModelStr.split(" "))
      calibrationWeights += weight.toDouble

    return Some(calibrationWeights.toArray)
  }

  def debugScore(example: Example, model: AbstractModel, transformer: Transformer) = {
    transformer.combineContextAndItems(example)
    for (ex <- example.example.asScala) {
      val builder = new java.lang.StringBuilder()
      model.debugScoreItem(example.example.get(0), builder)
      val result = builder.toString()
      println(result)
    }
  }

  def getLatestDirectory(dir: String): Option[String] = {
    val fs = FileSystem.get(new URI(dir), hadoopConfiguration)
    val path = new Path(dir)
    val files = fs.listStatus(path)
    if (files.isEmpty) {
      return None
    }
    val result = files
      .map(x => x.getPath().toString())
      .toBuffer
      .sortWith((a, b) => a > b)
      .head
    Some(result)
  }

  def writeStringToFile(str: String, output: String) = {
    val fs = FileSystem.get(new URI(output), hadoopConfiguration)
    val path = new Path(output)
    val stream = fs.create(path, true)
    val writer = new BufferedWriter(new OutputStreamWriter(stream))
    writer.write(str)
    writer.close()
  }

  def readStringFromFile(filename: String): String = {
    val fs = FileSystem.get(new URI(filename), hadoopConfiguration)
    val filePath = new Path(filename)
    if (!fs.exists(filePath)) {
      log.error(filename + " does not exist")
      return null
    }
    val fileStream = fs.open(filePath)
    val reader = new BufferedReader(new InputStreamReader(fileStream))
    val result = reader.readLine()
    reader.close()

    result
  }

  def trainAndSaveToFile(sc: SparkContext,
                         input: RDD[Example],
                         config: Config,
                         key: String) = {
    val trainer: String = config.getString(key + ".trainer")
    trainer match {
      case "linear" => LinearRankerTrainer.trainAndSaveToFile(sc, input, config, key)
      case "maxout" => MaxoutTrainer.trainAndSaveToFile(sc, input, config, key)
      case "spline" => SplineTrainer.trainAndSaveToFile(sc, input, config, key)
      case "boosted_stumps" => BoostedStumpsTrainer.trainAndSaveToFile(sc, input, config, key)
      case "decision_tree" => DecisionTreeTrainer.trainAndSaveToFile(sc, input, config, key)
      case "forest" => ForestTrainer.trainAndSaveToFile(sc, input, config, key)
      case "boosted_forest" => BoostedForestTrainer.trainAndSaveToFile(sc, input, config, key)
    }
  }

  // Returns distribution of features. Min, 1%tile, 99%tile, max
  def getFeatureDistribution(minCount : Int,
                             input : RDD[Example]) :
  Array[((String, String), (Double, Double, Double, Double))] = {
    case class Sample(min : Double, max : Double, count : Int, values : ArrayBuffer[Double])

    def updateSample(value : Double, sample : Sample) : Sample = {
      // Reservoir sample
      val buffer = sample.values
      if (buffer.size < 100) {
        buffer.append(value)
      } else {
        val rnd = scala.util.Random.nextInt(sample.count + 1)
        if (rnd < 100) {
          buffer(rnd) = value
        }
      }
      Sample(scala.math.min(value, sample.min),
             scala.math.max(value, sample.max),
             sample.count + 1,
             buffer)
    }

    input
      .mapPartitions(partition => {
      // family, feature name => min, max, count, sample
      val weights = new ConcurrentHashMap[(String, String), Sample]().asScala
      partition.foreach(example => {
        // Map(Feature family -> Map( feature name -> value ) )
        val flatFeature = Util.flattenFeature(example.example.get(0)).asScala
        flatFeature.foreach(familyMap => {
          familyMap._2.asScala.foreach(feature => {
            val key = (familyMap._1, feature._1)
            val curr = weights.getOrElse(key,
                                         Sample(Double.MaxValue, -Double.MaxValue, 0, ArrayBuffer[Double]()))
            val next = updateSample(feature._2, curr)
            weights.put(key, next)
          })
        })
      })
      weights.iterator
    })
      .reduceByKey((a, b) =>
                     Sample(scala.math.min(a.min, b.min),
                      scala.math.max(a.max, b.max),
                      a.count + b.count,
                      a.values ++ b.values))
      .filter(x => x._2.count >= minCount)
      .map(x => {
      val buffer = x._2.values.sortWith((a, b) => a < b)
      val one_percentile : Int = scala.math.min((buffer.size * 0.01).toInt, buffer.size - 1)
      val ninety_nine_percentile : Int = scala.math.min((buffer.size * 0.99).toInt, buffer.size - 1)
      (x._1, (x._2.min, buffer(one_percentile), buffer(ninety_nine_percentile), x._2.max))
    })
      .collect
      .toArray
  }
  
  def getLabel(fv : FeatureVector, rankKey : String, threshold : Double) : Double = {
    val rank = fv.floatFeatures.get(rankKey).asScala.head._2
    val label = if (rank <= threshold) {
      -1.0
    } else {
      1.0
    }
    return label
  }
}
