package com.airbnb.aerosolve.training

import java.io.BufferedWriter
import java.io.InputStreamReader
import java.io.BufferedReader
import java.io.OutputStreamWriter
import java.lang
import java.net.URI
import java.util.concurrent.ConcurrentHashMap

import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.core.FeatureVector
import com.airbnb.aerosolve.core.features._
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

import scala.collection.{Map, mutable, JavaConversions, JavaConverters}
import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import com.typesafe.config.Config

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
      case e : Throwable => log.error("Could not save model", e)
    }
  }

  def loadScoreModel(modelName: String, registry:FeatureRegistry): Option[AbstractModel] = {
    val fs = FileSystem.get(new URI(modelName), hadoopConfiguration)
    val modelPath = new Path(modelName)
    if (!fs.exists(modelPath)) {
      log.error(modelName + " does not exist")
      System.exit(-1)
    }
    val modelStream = fs.open(modelPath)
    val reader = new BufferedReader(new InputStreamReader(modelStream))
    val modelOpt = ModelFactory.createFromReader(reader, registry)
    if (!modelOpt.isPresent) {
      return None
    }
    val model = modelOpt.get()
    Some(model)
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

    Some(calibrationWeights.toArray)
  }

  def debugScore(example: Example, model: AbstractModel, transformer: Transformer) = {
    example.transform(transformer)
    for (vector <- example) {
      val builder = new java.lang.StringBuilder()
      model.debugScoreItem(vector, builder)
      val result = builder.toString
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
                         key: String,
                          registry: FeatureRegistry) = {
    val trainer: String = config.getString(key + ".trainer")
    trainer match {
      case "linear" => LinearRankerTrainer.trainAndSaveToFile(sc, input, config, key, registry)
      case "maxout" => MaxoutTrainer.trainAndSaveToFile(sc, input, config, key, registry)
      case "spline" => SplineTrainer.trainAndSaveToFile(sc, input, config, key, registry)
      case "boosted_stumps" => BoostedStumpsTrainer.trainAndSaveToFile(sc, input, config, key,
                                                                       registry)
      case "decision_tree" => DecisionTreeTrainer.trainAndSaveToFile(sc, input, config, key,
                                                                     registry)
      case "forest" => ForestTrainer.trainAndSaveToFile(sc, input, config, key,
                                                        registry)
      case "boosted_forest" => BoostedForestTrainer.trainAndSaveToFile(sc, input, config, key,
                                                                       registry)
      case "additive" => AdditiveModelTrainer.trainAndSaveToFile(sc, input, config, key,
                                                                 registry)
      case "kernel" => KernelTrainer.trainAndSaveToFile(sc, input, config, key,
                                                        registry)
      case "full_rank_linear" => FullRankLinearTrainer.trainAndSaveToFile(sc, input, config, key,
                                                                          registry)
      case "low_rank_linear" => LowRankLinearTrainer.trainAndSaveToFile(sc, input, config, key,
                                                                        registry)
      case "mlp" => MlpModelTrainer.trainAndSaveToFile(sc, input, config, key,
                                                       registry)
    }
  }
  
  def getLabel(fv : MultiFamilyVector, labelFamily : Family, threshold : Double) : Double = {
    // get label for classification
    val rank = getLabel(fv, labelFamily)
    if (rank <= threshold) {
      -1.0
    } else {
      1.0
    }
  }

  def getLabelDistribution(fv : MultiFamilyVector, labelFamily : Family) :
    Map[Feature, Double] = {
    JavaConversions.mapAsScalaMap(fv.get(labelFamily))
      .mapValues(d => d.doubleValue())
  }

  def getLabel(fv : MultiFamilyVector, labelFamily : Family) : Double = {
    // get label for regression
    // TODO (Brad): If we were confident about the feature name, this would be simpler.
    fv.get(labelFamily).iterator.next.value
  }

  // Returns the statistics of a feature
  case class FeatureStatistics(count : Double,min : Double, max : Double, mean : Double, variance : Double)

  def getFeatureStatistics(
                minCount : Int,
                input : RDD[Example]) : Array[(Feature, FeatureStatistics)] = {
    // ignore features present in less than minCount examples
    // output: Array[((featureFamily, featureName), (minValue, maxValue))]
    input
      .flatMap(example => example
        .flatMap((vector:java.lang.Iterable[FeatureValue]) => vector
          .map(fv => {
              val v = fv.value
              (fv.feature(), FeatureStatistics(1, v, v, v, v*v))
            })))
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
      .collect()
  }

  def getLabelCounts(minCount : Int,
                     input : RDD[Example],
                     labelFamily: Family) : Array[(Feature, Int)] = {
    input
      .mapPartitions(partition => {
        // family, feature name => count
        val weights = new ConcurrentHashMap[Feature, Int]().asScala
        partition.foreach(examples => {
          for (vector <- examples) {
            val labelVector = vector.get(labelFamily)
            if (labelVector != null) {
              for (fv <- labelVector.iterator) {
                val key = fv.feature
                val cur = weights.getOrElse(key, 0)
                weights.put(key, 1 + cur)
              }
            }
          }
        })
        weights.iterator
      })
      .reduceByKey((a, b) => a + b)
      .collect
  }

  def createStringDictionaryFromFeatureStatistics(stats : Array[(Feature, FeatureStatistics)],
                                                  excludedFamilies : Set[Family]) : StringDictionary = {
    val dictionary = new StringDictionary()
    for ((feature, featureStats) <- stats) {
      if (!excludedFamilies.contains(feature.family)) {
        if (featureStats.variance < 1e-6) {
          // Categorical feature, just pass through
          dictionary.possiblyAdd(feature, 0.0f, 1.0f)
        } else {
          val mean = featureStats.mean
          val scale = Math.sqrt(1.0 / featureStats.variance)
          dictionary.possiblyAdd(feature, mean, scale)
        }
      }
    }
    dictionary
  }

}
