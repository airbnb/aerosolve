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
    }
    TODO add rankKey: LABEL (default to GenericPipeline.LABEL)
  */
  def buildFeatureMapRun(sc: SparkContext, config : Config) = {
    val features = buildFeatures(sc, config)
    for (((family, name), feature) <- features) {
      // save to disk.
      if (feature.isInstanceOf[FeatureStats]) {
        val stats = feature.asInstanceOf[FeatureStats]
        log.info(s"${family}, ${name}: ${stats.count} ${stats.min} ${stats.max}")
      } else {
        val model = feature.asInstanceOf[NDTreeModel]
        val nodes = model.getNodes
        log.info(s"${family}, ${name}: ${model.getDimension} ${nodes.length} ${nodes.mkString("\n")}")
      }
    }
  }

  def buildFeatures(sc: SparkContext, config : Config):
      Array[((String, String), Any)] = {
    val cfg = config.getConfig("make_feature_map")
    val inputPattern: String = cfg.getString("input")
    log.info(s"Training data: ${inputPattern}")
    val sample: Double = cfg.getDouble("sample")
    val minCount: Int = cfg.getInt("min_count")
    // minMax.filter(x => !linearFeatureFamilies.contains(x._1._1))
    val linearFeatureFamilies: java.util.List[String] = Try(config.getStringList("linear_feature"))
      .getOrElse[java.util.List[String]](List.empty.asJava)

    val input = GenericPipeline.getExamples(sc, inputPattern)
    val options = NDTreeBuildOptions(
      maxTreeDepth = cfg.getInt("max_tree_depth"),
      minLeafCount = cfg.getInt("min_leaf_count"))

    val result: Array[((String, String), Any)] = getFeatures(
      sc, input, minCount, sample, linearFeatureFamilies, options)
    result
  }

  def getFeatures(sc: SparkContext, input: RDD[Example], minCount: Int, sample: Double,
                  linearFeatureFamilies: util.List[String],
                  options: NDTreeBuildOptions): Array[((String, String), Any)] = {
    val linearFeatureFamiliesBC = sc.broadcast(linearFeatureFamilies)
    val optionsBC = sc.broadcast(options)
    val tree: Array[((String, String), Any)] = input.sample(true, sample).mapPartitions(partition => {
      flattenExample(partition, linearFeatureFamiliesBC.value)
    }).reduceByKey((a, b) => {
      if (a.isInstanceOf[ArrayBuffer[Array[Double]]]) {
        a.asInstanceOf[ArrayBuffer[Any]].++=(b.asInstanceOf[ArrayBuffer[Any]])
      } else if (a.isInstanceOf[FeatureStats]) {
        val as = a.asInstanceOf[FeatureStats]
        val bs = b.asInstanceOf[FeatureStats]
        FeatureStats(as.count + bs.count,
          scala.math.min(as.min, bs.min),
          scala.math.max(as.max, bs.max))
      } else {
        log.error(s"reduceByKey unknow ${a}")
      }
    }).filter(x => {
      val a = x._2
      if (a.isInstanceOf[ArrayBuffer[Array[Double]]]) {
        a.asInstanceOf[ArrayBuffer[Any]].size >= minCount
      } else if (a.isInstanceOf[FeatureStats]) {
        a.asInstanceOf[FeatureStats].count >= minCount
      } else {
        log.error(s"filter unknown ${a}")
        false
      }
    }).map(x => {
      val a = x._2
      if (a.isInstanceOf[ArrayBuffer[Array[Double]]]) {
        // build tree
        ((x._1._1, x._1._2), NDTree(optionsBC.value,
          x._2.asInstanceOf[ArrayBuffer[Array[Double]]].toArray).model)
      } else {
        x
      }
    }).collect
    linearFeatureFamiliesBC.unpersist()
    optionsBC.unpersist()
    log.info(s"tree ${tree}")
    tree
  }

  def flattenExample(partition: Iterator[Example],
                     linearFeatureFamilies: java.util.List[String])
                  : Iterator[((String, String), Any)] = {
    val map = mutable.Map[(String, String), Any]()
    partition.foreach(examples => {
      examplesToFloatFeatureArray(examples, linearFeatureFamilies, map)
      examplesToDenseFeatureArray(examples, map)
      examplesToStringFeatureArray(examples, map)
    })
    map.iterator
  }

  def examplesToStringFeatureArray(example: Example, map: mutable.Map[(String, String), Any]): Unit = {
    for (i <- 0 until example.getExample.size()) {
      val featureVector = example.getExample.get(i)
      val stringFeatures = Util.safeMap(featureVector.stringFeatures).asScala
      stringFeatures.foreach(familyMap => {
        val family = familyMap._1

        familyMap._2.foreach(feature => {
          val key = (family, feature)
          val stats: FeatureStats =  map.getOrElseUpdate(key,
            FeatureStats(0, 1, 1)).asInstanceOf[FeatureStats]
          map.put(key, FeatureStats(stats.count + 1, 1, 1))
        })
      })
    }
  }

  def examplesToFloatFeatureArray(example: Example, linearFeatureFamilies:java.util.List[String],
                                  map: mutable.Map[(String, String), Any]): Unit = {
    for (i <- 0 until example.getExample.size()) {
      val featureVector = example.getExample.get(i)
      val floatFeatures = Util.safeMap(featureVector.floatFeatures).asScala
      floatFeatures.foreach(familyMap => {
        val family = familyMap._1
        if (linearFeatureFamilies.contains(family)) {
          familyMap._2.foreach(feature => {
            val key = (family, feature._1)
            val stats: FeatureStats =  map.getOrElseUpdate(key,
              FeatureStats(0, Double.MaxValue, Double.MinValue)).asInstanceOf[FeatureStats]
            val v = feature._2
            map.put(key, FeatureStats(stats.count + 1,
              scala.math.min(stats.min, v),
              scala.math.max(stats.max, v)))
          })
        } else if (family.compare(GenericPipeline.LABEL) != 0) {
          familyMap._2.foreach(feature => {
            // dense feature should not be same family and feature name as float
            val values:ArrayBuffer[Array[Double]] = map.getOrElseUpdate((family, feature._1),
              ArrayBuffer[Array[Double]]()).asInstanceOf[ArrayBuffer[Array[Double]]]
            values += Array[Double](feature._2.doubleValue())
          })
        }
      })
    }
  }

  def examplesToDenseFeatureArray(example: Example,
                                  map: mutable.Map[(String, String), Any]): Unit = {
    for (i <- 0 until example.getExample.size()) {
      val featureVector = example.getExample.get(i)
      val denseFeatures = Util.safeMap(featureVector.denseFeatures).asScala
      denseFeatures.foreach(feature => {
        val name = feature._1
        val values:ArrayBuffer[Array[Double]] =
          map.getOrElseUpdate((AdditiveModel.DENSE_FAMILY, name),
            ArrayBuffer[Array[Double]]()).asInstanceOf[ArrayBuffer[Array[Double]]]
        values += feature._2.asScala.toArray.map(_.doubleValue)
      })
    }
  }
}
