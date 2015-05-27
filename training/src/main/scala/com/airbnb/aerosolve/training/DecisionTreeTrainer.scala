package com.airbnb.aerosolve.training

import java.util

import com.airbnb.aerosolve.core.models.DecisionTreeModel
import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.core.ModelRecord
import com.airbnb.aerosolve.core.util.Util
import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

// The decision tree is meant to be a prior for the spline model / linear model
object DecisionTreeTrainer {
  private final val log: Logger = LoggerFactory.getLogger("DecisionTreeTrainer")

  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String) : DecisionTreeModel = {
    val candidateSize : Int = config.getInt(key + ".num_candidates")
    val rankKey : String = config.getString(key + ".rank_key")
    val maxDepth : Int = config.getInt(key + ".max_depth")

    val pointwise : RDD[Example] =
      LinearRankerUtils
        .makePointwiseFloat(input, config, key)

    val model = new DecisionTreeModel()
    val stumps = new util.ArrayList[ModelRecord]()
    model.setStumps(stumps)
    for (i <- 0 until maxDepth) {
      val modelBC = sc.broadcast(model)
      val newNodes = pointwise
        .map(x => Util.flattenFeature(x.example(0)))
        .map(x => (modelBC.value.getLeafIndex(x), x))
        .groupByKey
    }
    model
  }

  def getCandidateStumps(pointwise : Seq[Example],
                         candidateSize : Int,
                         rankKey : String) : Array[ModelRecord] = {
    val result = collection.mutable.HashSet[ModelRecord]()
    pointwise
      .flatMap(x => Util.flattenFeature(x.example(0)))
      .filter(x => x._1 != rankKey)
      .flatMap(x => {
        val buffer = collection.mutable.HashMap[(String, String), Double]()
        x._2.foreach(feature => {
          buffer.put((x._1, feature._1), feature._2)
        })
        buffer
      })
    .take(candidateSize)
    .foreach(x => {
      val rec = new ModelRecord()
      rec.setFeatureFamily(x._1._1)
      rec.setFeatureName(x._1._2)
      rec.setThreshold(x._2)
      result.add(rec)
    })
    result.toArray
  }

  def trainAndSaveToFile(sc : SparkContext,
                         input : RDD[Example],
                         config : Config,
                         key : String) = {
    val model = train(sc, input, config, key)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }
}
