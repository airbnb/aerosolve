package com.airbnb.aerosolve.training.pipeline

import java.util

import com.airbnb.aerosolve.core.models.AdditiveModel
import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.core.{Example, NDTreeNode}
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
                                  splineFeatureFamilies: util.List[String],
                                  checkPointDir: String,
                                  maxTreeDepth1Dimension: Int,
                                  maxTreeDepthPerDimension: Int,
                                  minLeafCount: Int,
                                  minLeafCountPercent: Double,
                                  minLeafValuePercent: Double)

  case class FeatureStats(count: Double, min: Double, max: Double,
                          spline: Boolean = false)
  /*
    build NDTree from examples, each float/dense feature generates a NDTree
    and save to FeatureMap
    Sample config
    make_feature_map {
      input :  ${training_data}
      output:  ${feature_map}
      sample: 0.01
      min_count: 200
      // max_tree_depth = max(max_tree_depth_1_dimension + 1, max_tree_depth_per_dimension * dimension + 1)
      max_tree_depth_1_dimension: 6  ( max nodes could be 2^(max_tree_depth_1_dimension) )
      max_tree_depth_per_dimension: 4( max nodes could be 2^(max_tree_depth_per_dimension * dimension) )
      // default 1/2^(max single dimension depth(max_tree_depth_1_dimension or max_tree_depth_per_dimension) )
      // if you set it to 0, it will be replaced with default value too.
      min_leaf_value_percent: 0.01
       // if you set it to 0, default value is 1/2^(max_tree_depth -1)
      min_leaf_count_percent: 0.01
      // the actual min_leaf_count = max(min_leaf_count, min_leaf_count_percent * total count)
      min_leaf_count: 20
      // feature families in linear_feature should use linear
      linear_feature: ["L", "T", "L_x_T"]
      // set feature families using spline
      spline_feature: ["ds"]
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
        case Left(nodes) => {
          log.info(s"${family}, ${name}: ${nodes.length} ${nodes.mkString("\n")}")
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
    val splineFeatureFamilies: java.util.List[String] = Try(
      cfg.getStringList("spline_feature"))
      .getOrElse[java.util.List[String]](List.empty.asJava)
    val checkPointDir = Try(cfg.getString("check_point_dir")).getOrElse("")
    val maxTreeDepth1Dimension = cfg.getInt("max_tree_depth_1_dimension")
    val maxTreeDepthPerDimension = cfg.getInt("max_tree_depth_per_dimension")
    val minLeafCountPercent: Double = Try(cfg.getDouble("min_leaf_count_percent")).getOrElse(0)
    val minLeafValuePercent: Double = Try(cfg.getDouble("min_leaf_value_percent")).getOrElse(0)
    val minLeafCount: Int = Try(cfg.getInt("min_leaf_count")).getOrElse(0)
    NDTreePipelineParams(
      cfg.getDouble("sample"),
      cfg.getInt("min_count"),
      linearFeatureFamilies,
      splineFeatureFamilies,
      checkPointDir,
      maxTreeDepth1Dimension,
      maxTreeDepthPerDimension,
      minLeafCount,
      minLeafCountPercent,
      minLeafValuePercent)
  }

  def buildFeatures(sc: SparkContext, config : Config):
      Array[((String, String), Either[Array[NDTreeNode], FeatureStats])] = {
    val cfg = config.getConfig("make_feature_map")
    val inputPattern: String = cfg.getString("input")
    val input = GenericPipeline.getExamples(sc, inputPattern)
    log.info(s"Training data: ${inputPattern}")
    val params: NDTreePipelineParams = getNDTreePipelineParams(cfg)

    val result: Array[((String, String), Either[Array[NDTreeNode], FeatureStats])] = getFeatures(
      sc, input.sample(false, params.sample), params)
    result
  }

  def getFeatures(sc: SparkContext, input: RDD[Example], params: NDTreePipelineParams):
                  Array[((String, String), Either[Array[NDTreeNode], FeatureStats])] = {
    val featureRDD: RDD[((String, String), Either[ArrayBuffer[Array[Double]], FeatureStats])] =
      input.mapPartitions(partition => {
        flattenExample(partition, params)
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
              scala.math.max(y.max, bs.max), y.spline))
          }
        }
        result
      }).filter(x => {
        val a = x._2
        a match {
          case Left(x) => {
            x.size >= params.minCount
          }
          case Right(y) => {
            y.count >= params.minCount
          }
        }
      })
    if (!params.checkPointDir.isEmpty) {
      sc.setCheckpointDir(params.checkPointDir)
      featureRDD.checkpoint()
    }

    // featureRDD cached so that we don't rerun the previous steps
    val tree: Array[((String, String), Either[Array[NDTreeNode], FeatureStats])] =
      featureRDD.map(x => {
        val a = x._2
        val result: ((String, String), Either[Array[NDTreeNode], FeatureStats]) = a match {
          case Left(y) => {
            // build tree
            // FloatToDense transform make sure all array in y are same size.
            // so we just pick first one to get the dimension
            val dimension = y(0).length

            val minLeafValuePercent: Double = if (params.minLeafValuePercent != 0) {
              params.minLeafValuePercent
            } else if (dimension == 1) {
              1.0/scala.math.pow(2, params.maxTreeDepth1Dimension + 1)
            } else {
              1.0/scala.math.pow(2, params.maxTreeDepthPerDimension + 1)
            }

            val minLeafCountPercent: Double = if (params.minLeafCountPercent != 0) {
              params.minLeafCountPercent
            } else if (dimension == 1) {
              1.0/scala.math.pow(2, params.maxTreeDepth1Dimension + 1)
            } else {
              1.0/scala.math.pow(2, params.maxTreeDepthPerDimension * dimension + 1)
            }

            val options = NDTreeBuildOptions(
              math.max(params.maxTreeDepth1Dimension + 1,
                params.maxTreeDepthPerDimension * dimension + 1),
                math.max(params.minLeafCount, (minLeafCountPercent * y.length).toInt),
                minLeafValuePercent)

              ((x._1._1, x._1._2), Left(NDTree.buildTree(options, y.toArray)))
          }
          case Right(z) => {
            ((x._1._1, x._1._2), Right(z))
          }
        }
        result
    }).collect

    log.info(s"tree length ${tree.length}")
    tree
  }

  def flattenExample(partition: Iterator[Example],
                     params: NDTreePipelineParams)
                  : Iterator[((String, String), Either[ArrayBuffer[Array[Double]], FeatureStats])] = {
    val map = mutable.Map[(String, String), Either[ArrayBuffer[Array[Double]], FeatureStats]]()
    partition.foreach(examples => {
      examplesToFloatFeatureArray(examples, params, map)
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

  def examplesToFloatFeatureArray(example: Example,
                                  params: NDTreePipelineParams,
                                  map: mutable.Map[(String, String),
                                  Either[ArrayBuffer[Array[Double]], FeatureStats]]): Unit = {
    val linearFeatureFamilies = params.linearFeatureFamilies
    val splineFeatureFamilies = params.splineFeatureFamilies
    for (i <- 0 until example.getExample.size()) {
      val featureVector = example.getExample.get(i)
      val floatFeatures = Util.safeMap(featureVector.floatFeatures).asScala
      floatFeatures.foreach(familyMap => {
        val family = familyMap._1
        val isLinear = linearFeatureFamilies.contains(family)
        val isSpline = splineFeatureFamilies.contains(family)
        if (isLinear || isSpline) {
          familyMap._2.foreach(feature => {
            val key = (family, feature._1)
            val stats =  map.getOrElseUpdate(key,
              Right(FeatureStats(0, Double.MaxValue, Double.MinValue))).right.get
            val v = feature._2
            map.put(key, Right(FeatureStats(
              stats.count + 1,
              scala.math.min(stats.min, v),
              scala.math.max(stats.max, v),
              isSpline)))
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
