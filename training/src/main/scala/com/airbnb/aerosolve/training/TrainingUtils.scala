package com.airbnb.aerosolve.training

import java.io.{BufferedReader, BufferedWriter, InputStreamReader, OutputStreamWriter}
import java.net.URI
import java.util.concurrent.ConcurrentHashMap

import com.airbnb.aerosolve.core.models.{AbstractModel, ModelFactory}
import com.airbnb.aerosolve.core.transforms.Transformer
import com.airbnb.aerosolve.core.util.{StringDictionary, Util}
import com.airbnb.aerosolve.core.{Example, FeatureVector}
import com.typesafe.config.Config
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.hive.HiveContext
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

object TrainingUtils {

  val log: Logger = LoggerFactory.getLogger("TrainingUtils")
  val hadoopConfiguration = new Configuration()

  def downsample(input: RDD[Example],
                 loss: String,
                 rankKey: String,
                 threshold: Double,
                 downsample: Map[Int, Float]): RDD[Example] = {
    val sample = input.mapPartitions(examples => {
      val rnd = new java.util.Random()
      examples.flatMap { e =>
        val label = getLabel(e, loss, rankKey, threshold).toInt
        if (downsample.contains(label)) {
          val w = downsample(label)
          if (rnd.nextDouble() <= w) {
            Some(e)
          } else {
            None
          }
        } else {
          Some(e)
        }
      }
    })
    sample
  }

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

  def saveModel(model: AbstractModel,
                config: Config,
                key: String): Unit = {
    try {
      val output = config.getString(key)
      saveModel(model, output)
    } catch {
      case _: Throwable => log.error("Could not save model")
    }
  }

  def saveModel(model: AbstractModel, output: String): Unit = {
    try {
      val fileSystem = FileSystem.get(new java.net.URI(output), new Configuration())
      val file = fileSystem.create(new Path(output), true)
      val writer = new BufferedWriter(new OutputStreamWriter(file))
      model.save(writer)
      writer.close()
      file.close()
      log.info(s"Saved model to $output")
    } catch {
      case e: Throwable => log.error(s"Could not save model at $output because $e")
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
    Option(modelOpt.orNull())
  }

  def loadPlattScaleWeights(filePath: String): Option[Array[Double]] = {
    val calibrationModelStr =
      TrainingUtils.readStringFromFile(filePath)
    if (calibrationModelStr == null) {
      // error already logged in readStringFromFile
      None
    } else {
      val calibrationWeights = scala.collection.mutable.ArrayBuffer.empty[Double]
      for (weight <- calibrationModelStr.split(" "))
        calibrationWeights += weight.toDouble

      Some(calibrationWeights.toArray)
    }
  }

  def debugScore(example: Example, model: AbstractModel, transformer: Transformer) = {
    transformer.combineContextAndItems(example)
    for (ex <- example.example.asScala) {
      val builder = new java.lang.StringBuilder()
      model.debugScoreItem(example.example.get(0), builder)
      val result = builder.toString
      println(result)
    }
  }

  def getLatestDirectory(dir: String): Option[String] = {
    val fs = FileSystem.get(new URI(dir), hadoopConfiguration)
    val path = new Path(dir)
    val files = fs.listStatus(path).toSeq
    files
      .map(x => x.getPath.toString)
      .sortWith((a, b) => a > b)
      .headOption
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
      null
    } else {
      val fileStream = fs.open(filePath)
      val reader = new BufferedReader(new InputStreamReader(fileStream))
      val result = reader.readLine()
      reader.close()

      result
    }
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

  def getLabel(example: Example, loss: String, rankKey: String, threshold: Double): Double = {
    val fv = example.getExample.get(0)
    val label: Double = if (loss == "regression") {
      TrainingUtils.getLabel(fv, rankKey)
    } else {
      TrainingUtils.getLabel(fv, rankKey, threshold)
    }
    label
  }

  def getLabel(fv: FeatureVector, rankKey: String, threshold: Double): Double = {
    // get label for classification
    val rank = fv.floatFeatures.get(rankKey).values().iterator().next().doubleValue()
    if (rank <= threshold) {
      -1.0
    } else {
      1.0
    }
  }

  def getLabelDistribution(fv: FeatureVector, rankKey: String): Map[String, Double] = {
    fv.floatFeatures.get(rankKey).asScala.map(x => (x._1.toString, x._2.toDouble)).toMap
  }

  def getLabel(fv: FeatureVector, rankKey: String): Double = {
    // get label for regression
    fv.floatFeatures.get(rankKey).asScala.head._2.toDouble
  }

  // Returns the statistics of a feature
  case class FeatureStatistics(
                                count: Double,
                                min: Double,
                                max: Double,
                                mean: Double,
                                variance: Double,
                                quantiles: Seq[Double]
                              )

  /**
    * Compute [[FeatureStatistics]] across examples and ignore those with less than minCount features.
    */
  def getFeatureStatistics(minCount: Int,
                           input: RDD[Example],
                           quantiles: Seq[Double] = Nil): Array[((String, String), FeatureStatistics)] = {
    // TODO: use unified SparkSession for Spark 2.0+ as we drop deps on Hive UDAF
    val sqlContext = if (quantiles.isEmpty) {
      SQLContext.getOrCreate(input.sparkContext)
    } else {
      new HiveContext(input.sparkContext)
    }
    import sqlContext.implicits._
    import org.apache.spark.sql.functions._

    // number of statistics we always extract regardless if we ask for quantiles
    val NUM_BASIC_STATS = 7

    input
      .flatMap(examples => {
        examples.example.iterator().flatMap {
          example =>
            Util.flattenFeatureAsStream(example).iterator()
              .flatMap {
                featureFamily =>
                  val family = featureFamily.getKey
                  featureFamily.getValue.iterator().map {
                    feature =>
                      val value = feature.getValue.toDouble
                      (family, feature.getKey, value)
                  }
              }
        }
      })
      .toDF("family", "feature", "value")
      .groupBy($"family", $"feature")
      .agg(
        count(lit(1)) as "count",
        Seq(
          min($"value"),
          max($"value"),
          sum($"value"),
          // TODO: use built-in stdev function in Spark 1.6
          sum($"value" * $"value"),
          // TODO: use built-in percentileApprox in Spark 2.0
          if (quantiles.isEmpty) lit(null) else callUDF("percentile_approx", $"value", array(quantiles.map(lit): _*))
        ): _*
      )
      // ignore features present in less than minCount examples
      .where($"count" >= minCount)
      .map {
        row =>
          val Seq(family: String, feature: String, count: Long, min: Double, max: Double, sum: Double, sqSum: Double) = row.toSeq.take(NUM_BASIC_STATS)
          val values = if (row.isNullAt(NUM_BASIC_STATS)) Nil else row.getAs[Seq[Double]](NUM_BASIC_STATS)

          ((family, feature),
            FeatureStatistics(
              count,
              min,
              max,
              sum / count,
              (sqSum - sum * sum / count) / (count - 1),
              values
            ))
      }
      .collect
  }

  def getLabelCounts(minCount: Int,
                     input: RDD[Example],
                     rankKey: String): Array[((String, String), Int)] = {
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

  def createStringDictionaryFromFeatureStatistics(stats: Array[((String, String), FeatureStatistics)],
                                                  excludedFamilies: Set[String]): StringDictionary = {
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
