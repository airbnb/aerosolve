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
import com.airbnb.aerosolve.core.util.StringDictionary
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
import scala.collection.JavaConversions._
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
                key : String): Unit = {
    try {
      val output = config.getString(key)
      saveModel(model, output)
    } catch {
      case _ : Throwable => log.error("Could not save model")
    }
  }

  def saveModel(model: AbstractModel, output: String): Unit = {
    try {
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
      case "additive" => AdditiveModelTrainer.trainAndSaveToFile(sc, input, config, key)
      case "kernel" => KernelTrainer.trainAndSaveToFile(sc, input, config, key)
      case "full_rank_linear" => FullRankLinearTrainer.trainAndSaveToFile(sc, input, config, key)
      case "low_rank_linear" => LowRankLinearTrainer.trainAndSaveToFile(sc, input, config, key)
      case "mlp" => MlpModelTrainer.trainAndSaveToFile(sc, input, config, key)
    }
  }
  
  def getLabel(fv : FeatureVector, rankKey : String, threshold : Double) : Double = {
    // get label for classification
    val rank = fv.floatFeatures.get(rankKey).asScala.head._2
    val label = if (rank <= threshold) {
      -1.0
    } else {
      1.0
    }
    return label
  }

  def getLabelDistribution(fv : FeatureVector, rankKey : String) : Map[String, Double] = {
    fv.floatFeatures.get(rankKey).asScala.map(x => (x._1.toString, x._2.toDouble)).toMap
  }

  def getLabel(fv : FeatureVector, rankKey : String) : Double = {
    // get label for regression
    fv.floatFeatures.get(rankKey).asScala.head._2.toDouble
  }

  // Returns the statistics of a feature
  case class FeatureStatistics(count : Double,min : Double, max : Double, mean : Double, variance : Double)

  def getFeatureStatistics(
                minCount : Int,
                input : RDD[Example]) : Array[((String, String), FeatureStatistics)] = {
    // ignore features present in less than minCount examples
    // output: Array[((featureFamily, featureName), (minValue, maxValue))]
    input
      .mapPartitions(partition => {
      // family, feature name => count, min, max, sum x, sum x ^ 2
      val weights = new ConcurrentHashMap[(String, String), FeatureStatistics]().asScala
      partition.foreach(examples => {
        for (i <- 0 until examples.example.size()) {
          val flatFeature = Util.flattenFeature(examples.example.get(i)).asScala
          flatFeature.foreach(familyMap => {
            familyMap._2.foreach(feature => {
              val key = (familyMap._1, feature._1)
              val curr = weights.getOrElse(key, FeatureStatistics(0, Double.MaxValue, -Double.MaxValue, 0.0, 0.0))
                val v = feature._2
                weights.put(key,
                            FeatureStatistics(curr.count + 1,
                             scala.math.min(curr.min, v),
                             scala.math.max(curr.max, v),
                             curr.mean + v, // actually the sum
                             curr.variance + v * v) // actually the sum of squares
                )
              })
          })
        }
      })
      weights.iterator
    })
      .reduceByKey((a, b) =>
                     FeatureStatistics(a.count + b.count,
                      scala.math.min(a.min, b.min),
                      scala.math.max(a.max, b.max),
                      a.mean + b.mean,
                      a.variance + b.variance))
      .filter(x => x._2.count >= minCount)
      .map(x => (x._1,
          FeatureStatistics(
           count = x._2.count,
           min = x._2.min,
           max = x._2.max,
           mean = x._2.mean / x._2.count,
           variance = (x._2.variance - x._2.mean * x._2.mean / x._2.count) / (x._2.count - 1.0)
           )))
      .collect
  }

  def getLabelCounts(minCount : Int,
                     input : RDD[Example],
                     rankKey: String) : Array[((String, String), Int)] = {
    input
      .mapPartitions(partition => {
        // family, feature name => count
        val weights = new ConcurrentHashMap[(String, String), Int]().asScala
        partition.foreach(examples => {
          for (i <- 0 until examples.example.size()) {
            val example = examples.example.get(i)
            val floatFeatures = example.getFloatFeatures
            val stringFeatures = example.getStringFeatures
            if (floatFeatures.containsKey(rankKey)) {
              for (labelEntry <- floatFeatures.get(rankKey)) {
                val key = (rankKey, labelEntry._1)
                val cur = weights.getOrElse(key, 0)
                weights.put(key, 1 + cur)
              }
            } else if (stringFeatures.containsKey(rankKey)) {
              for (labelName <- stringFeatures.get(rankKey)) {
                val key = (rankKey, labelName)
                val cur = weights.getOrElse(key, 0)
                weights.put(key, 1 + cur)
              }
            }
          }
        }
        )
        weights.iterator
      })
      .reduceByKey((a, b) => a + b)
      .collect
  }

  def createStringDictionaryFromFeatureStatistics(stats : Array[((String, String), FeatureStatistics)],
                                                  excludedFamilies : Set[String]) : StringDictionary = {
    val dictionary = new StringDictionary()
    for (stat <- stats) {
      val (family, feature) = stat._1
      if (!excludedFamilies.contains(family)) {
        if (stat._2.variance < 1e-6) {
          // Categorical feature, just pass through
          dictionary.possiblyAdd(family, feature, 0.0f, 1.0f)
        } else {
          val mean = stat._2.mean
          val scale = Math.sqrt(1.0 / stat._2.variance)
          dictionary.possiblyAdd(family, feature, mean, scale)
        }
      }
    }
    dictionary
  }

}
