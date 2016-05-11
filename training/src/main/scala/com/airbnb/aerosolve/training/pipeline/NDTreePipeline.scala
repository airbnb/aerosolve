package com.airbnb.aerosolve.training.pipeline

import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.core.models.AdditiveModel
import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.training.NDTree
import com.airbnb.aerosolve.training.NDTree.NDTreeBuildOptions
import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Try

object NDTreePipeline {
  val log: Logger = LoggerFactory.getLogger("NDTreePipeline")
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
  */
  def buildFeatureMapRun(sc: SparkContext, config : Config) = {
    val cfg = config.getConfig("make_feature_map")
    val inputPattern: String = cfg.getString("input")
    val sample: Double = cfg.getDouble("sample")
    val minCount: Int = config.getInt("min_count")
    // minMax.filter(x => !linearFeatureFamilies.contains(x._1._1))
    val linearFeatureFamilies: java.util.List[String] = Try(config.getStringList("linear_feature"))
      .getOrElse[java.util.List[String]](List.empty.asJava)

    val linearFeatureFamiliesBC = sc.broadcast(linearFeatureFamilies)
    val options = sc.broadcast(NDTreeBuildOptions(
      maxTreeDepth = config.getInt("max_tree_depth"),
      minLeafCount = config.getInt("min_leaf_count")))

    log.info("Training data: ${inputPattern}")
    val input = GenericPipeline.getExamples(sc, inputPattern).sample(true, sample)
    val tree = input.mapPartitions(partition => {
      flattenExample(partition, linearFeatureFamiliesBC.value)
    }).reduceByKey((a, b) => {
      a.++=(b)
    }).filter(x => x._2.size >= minCount)
      // build tree
      .map(x => ((x._1._1, x._1._2), NDTree(options.value, x._2.toArray).nodes))
      .collect
    for (((featureFamily, featureName), features) <- tree) {
      // save to disk.

    }
  }

  def flattenExample(partition: Iterator[Example],
                  linearFeatureFamilies: java.util.List[String])
                  : Iterator[((String, String), ArrayBuffer[Array[Double]])] = {
    val map = mutable.Map[(String, String), ArrayBuffer[Array[Double]]]()
    partition.foreach(examples => {
      examplesToFloatFeatureArray(examples, linearFeatureFamilies, map)
      examplesToDenseFeatureArray(examples, map)
    })
    map.iterator
  }

  def examplesToFloatFeatureArray(example: Example, linearFeatureFamilies:java.util.List[String],
                                  map: mutable.Map[(String, String), ArrayBuffer[Array[Double]]]): Unit = {
    for (i <- 0 until example.getExample.size()) {
      val featureVector = example.getExample.get(i)
      val floatFeatures = Util.safeMap(featureVector.floatFeatures).asScala
      floatFeatures.foreach(familyMap => {
        val family = familyMap._1
        if (!linearFeatureFamilies.contains(family)) {
          familyMap._2.foreach(feature => {
            // dense feature should not be same family and feature name as float
            val values:ArrayBuffer[Array[Double]] = map.getOrElseUpdate((family, feature._1),
              ArrayBuffer[Array[Double]]())
            values += Array[Double](feature._2.doubleValue())
          })
        }
      })
    }
  }

  def examplesToDenseFeatureArray(example: Example,
                                  map: mutable.Map[(String, String), ArrayBuffer[Array[Double]]]): Unit = {
    for (i <- 0 until example.getExample.size()) {
      val featureVector = example.getExample.get(i)
      val denseFeatures = Util.safeMap(featureVector.denseFeatures).asScala
      denseFeatures.foreach(feature => {
        val name = feature._1
        val values:ArrayBuffer[Array[Double]] =
          map.getOrElseUpdate((AdditiveModel.DENSE_FAMILY, name),
            ArrayBuffer[Array[Double]]())
        values += feature._2.asScala.toArray.map(_.doubleValue)
      })
    }
  }
}
