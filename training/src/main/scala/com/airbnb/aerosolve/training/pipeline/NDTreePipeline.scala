package com.airbnb.aerosolve.training.pipeline

import java.util

import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.core.models.{AdditiveModel, NDTreeModel}
import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.training.NDTree
import com.airbnb.aerosolve.training.NDTree.NDTreeBuildOptions
import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Try

object NDTreePipeline {
  val log: Logger = LoggerFactory.getLogger("NDTreePipeline")

  case class NDTreePipelineParams(sample: Double,
                                  minCount: Int,
                                  linearFeatureFamilies: util.List[String],
                                  checkPointDir: String,
                                  options: NDTreeBuildOptions)

  case class FeatureStats(count: Double, min: Double, max: Double)
  /*
    build NDTree from examples, each float/dense feature generates a NDTree
    and save to FeatureMap
    Sample config
    make_feature_map {
      input :  ${training_data}
      output:  ${feature_map}
      sample: 0.01
      min_count: 200
      max_tree_depth: 7  ( max nodes could be 2^(max_tree_depth-1) )
      min_leaf_count: 200
      // feature families in linear_feature should use linear
      linear_feature: ["L", "T", "L_x_T"]
      // if your job failed, try to set check_point to avoid rerun
      // and set("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
      // clean checkpoint files if the reference is out of scope.
      check_point_dir: "hdfs://server/team/project/tmp"
    }
    TODO add rankKey: LABEL (default to GenericPipeline.LABEL)
  */
  def buildFeatureMapRun(sc: SparkContext, config : Config) = {
    val features = buildFeatures(sc, config)
    for (((family, name), feature) <- features) {
      // TODO save to disk.
      feature match {
        case Left(model) => {
          val nodes = model.getNodes
          log.info(s"${family}, ${name}: ${model.getDimension} ${nodes.length} ${nodes.mkString("\n")}")
        }
        case Right(stats) => {
          log.info(s"${family}, ${name}: ${stats.count} ${stats.min} ${stats.max}")
        }
      }
    }
  }

  def getNDTreePipelineParams(cfg: Config): NDTreePipelineParams = {
    val linearFeatureFamilies: java.util.List[String] = Try(cfg.getStringList("linear_feature"))
      .getOrElse[java.util.List[String]](List.empty.asJava)
    val checkPointDir = Try(cfg.getString("check_point_dir")).getOrElse("")
    val options = NDTreeBuildOptions(
      maxTreeDepth = cfg.getInt("max_tree_depth"),
      minLeafCount = cfg.getInt("min_leaf_count"))
    NDTreePipelineParams(
      cfg.getDouble("sample"),
      cfg.getInt("min_count"),
      linearFeatureFamilies,
      checkPointDir,
      options
    )
  }

  def buildFeatures(sc: SparkContext, config : Config):
      Array[((String, String), Either[NDTreeModel, FeatureStats])] = {
    val cfg = config.getConfig("make_feature_map")
    val inputPattern: String = cfg.getString("input")
    val input = GenericPipeline.getExamples(sc, inputPattern)
    log.info(s"Training data: ${inputPattern}")
    val params: NDTreePipelineParams = getNDTreePipelineParams(cfg)

    val result: Array[((String, String), Either[NDTreeModel, FeatureStats])] = getFeatures(
      sc, input.sample(false, params.sample), params)
    result
  }

  def getFeatures(sc: SparkContext, input: RDD[Example], params: NDTreePipelineParams):
                  Array[((String, String), Either[NDTreeModel, FeatureStats])] = {
    val paramsBC = sc.broadcast(params)
    val featureRDD: RDD[((String, String), Either[ArrayBuffer[Array[Double]], FeatureStats])] =
      input.mapPartitions(partition => {
        flattenExample(partition, paramsBC.value.linearFeatureFamilies)
    }).reduceByKey((a, b) => {
        val result = a match {
          case Left(x) => {
            x.++=(b.left.get)
            a
          }
          case Right(y) => {
            val bs = b.right.get
            Right(FeatureStats(y.count + bs.count,
              scala.math.min(y.min, bs.min),
              scala.math.max(y.max, bs.max)))
          }
        }
        result
      }).filter(x => {
        val a = x._2
        a match {
          case Left(x) => {
            x.size >= paramsBC.value.minCount
          }
          case Right(y) => {
            y.count >= paramsBC.value.minCount
          }
        }
      })
    if (!paramsBC.value.checkPointDir.isEmpty) {
      sc.setCheckpointDir(paramsBC.value.checkPointDir)
      featureRDD.checkpoint()
    }

    // featureRDD cached so that we don't rerun the previous steps
    val tree: Array[((String, String), Either[NDTreeModel, FeatureStats])] =
      featureRDD.map(x => {
        val a = x._2
        val result: ((String, String), Either[NDTreeModel, FeatureStats]) = a match {
          case Left(y) => {
            // build tree
            ((x._1._1, x._1._2), Left(NDTree(paramsBC.value.options, y.toArray).model))
          }
          case Right(z) => {
            ((x._1._1, x._1._2), Right(z))
          }
        }
        result
    }).collect
    paramsBC.unpersist()

    log.info(s"tree length ${tree.length}")
    tree
  }

  def flattenExample(partition: Iterator[Example],
                     linearFeatureFamilies: java.util.List[String])
                  : Iterator[((String, String), Either[ArrayBuffer[Array[Double]], FeatureStats])] = {
    val map = mutable.Map[(String, String), Either[ArrayBuffer[Array[Double]], FeatureStats]]()
    partition.foreach(examples => {
      examplesToFloatFeatureArray(examples, linearFeatureFamilies, map)
      examplesToDenseFeatureArray(examples, map)
      examplesToStringFeatureArray(examples, map)
    })
    map.iterator
  }

  def examplesToStringFeatureArray(example: Example, map: mutable.Map[(String, String),
      Either[ArrayBuffer[Array[Double]], FeatureStats]]): Unit = {
    for (i <- 0 until example.getExample.size()) {
      val featureVector = example.getExample.get(i)
      val stringFeatures = Util.safeMap(featureVector.stringFeatures).asScala
      stringFeatures.foreach(familyMap => {
        val family = familyMap._1
        familyMap._2.foreach(feature => {
          val key = (family, feature)
          val stats =  map.getOrElseUpdate(key,
            Right(FeatureStats(0, 1, 1))).right.get
          map.put(key, Right(FeatureStats(stats.count + 1, 1, 1)))
        })
      })
    }
  }

  def examplesToFloatFeatureArray(example: Example, linearFeatureFamilies:java.util.List[String],
                                  map: mutable.Map[(String, String),
                                    Either[ArrayBuffer[Array[Double]], FeatureStats]]): Unit = {
    for (i <- 0 until example.getExample.size()) {
      val featureVector = example.getExample.get(i)
      val floatFeatures = Util.safeMap(featureVector.floatFeatures).asScala
      floatFeatures.foreach(familyMap => {
        val family = familyMap._1
        if (linearFeatureFamilies.contains(family)) {
          familyMap._2.foreach(feature => {
            val key = (family, feature._1)
            val stats =  map.getOrElseUpdate(key,
              Right(FeatureStats(0, Double.MaxValue, Double.MinValue))).right.get
            val v = feature._2
            map.put(key, Right(FeatureStats(stats.count + 1,
              scala.math.min(stats.min, v),
              scala.math.max(stats.max, v))))
          })
        } else if (family.compare(GenericPipeline.LABEL) != 0) {
          familyMap._2.foreach(feature => {
            // dense feature should not be same family and feature name as float
            val values:ArrayBuffer[Array[Double]] = map.getOrElseUpdate((family, feature._1),
              Left(ArrayBuffer[Array[Double]]())).left.get
            values += Array[Double](feature._2.doubleValue())
          })
        }
      })
    }
  }

  def examplesToDenseFeatureArray(example: Example,
                                  map: mutable.Map[(String, String),
                                    Either[ArrayBuffer[Array[Double]], FeatureStats]]): Unit = {
    for (i <- 0 until example.getExample.size()) {
      val featureVector = example.getExample.get(i)
      val denseFeatures = Util.safeMap(featureVector.denseFeatures).asScala
      denseFeatures.foreach(feature => {
        val name = feature._1
        val values:ArrayBuffer[Array[Double]] =
          map.getOrElseUpdate((AdditiveModel.DENSE_FAMILY, name),
            Left(ArrayBuffer[Array[Double]]())).left.get
        values += feature._2.asScala.toArray.map(_.doubleValue)
      })
    }
  }
}
