package com.airbnb.aerosolve.training

import java.util

import com.airbnb.aerosolve.core.models.BoostedStumpsModel
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

// The boosted stump model is meant to be a prior for the spline model
// Known issues - the transforms could be applied twice, once here and once in the linear trainer
// but most transforms will be empty / identity anyway since this is a sub model for the spline model.
object BoostedStumpsTrainer {
  private final val log: Logger = LoggerFactory.getLogger("BoostedStumpsTrainer")

  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String) : BoostedStumpsModel = {
    val candidateSize : Int = config.getInt(key + ".num_candidates")
    val rankKey : String = config.getString(key + ".rank_key")

    val pointwise : RDD[Example] =
      LinearRankerUtils
        .makePointwiseFloat(input, config, key)

    val candidates : Array[ModelRecord] = getCandidateStumps(pointwise, candidateSize, rankKey)

    val data : RDD[Example] = getResponses(sc, pointwise, candidates, rankKey).cache()

    val weights = LinearRankerTrainer.train(sc, data, config, key).toMap

    // Lookup each candidate's weights
    (0 until candidates.size).foreach(i => {
      val stump = candidates(i)
      val pos = weights.getOrElse(("+", i.toString), 0.0)
      stump.setFeatureWeight(pos)
    })

    val sorted = candidates.toBuffer.sortWith((a, b) => math.abs(a.featureWeight) > math.abs(b.featureWeight))

    val stumps = new util.ArrayList[ModelRecord]()
    sorted.foreach(stump => {
      if (math.abs(stump.featureWeight) > 0.0) {
        stumps.add(stump)
      }
    })

    val model = new BoostedStumpsModel()
    model.setStumps(stumps)
    model
  }

  def getCandidateStumps(pointwise : RDD[Example],
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

  def getResponses(sc : SparkContext,
                   pointwise : RDD[Example],
                   candidates : Array[ModelRecord],
                   rankKey : String) : RDD[Example] = {
    val candidatesBC = sc.broadcast(candidates)
    pointwise.map(example => {
      val cand  = candidatesBC.value
      val ex = Util.flattenFeature(example.example.get(0))
      val output = new Example()
      val fv = Util.createNewFeatureVector()
      output.addToExample(fv)
      fv.floatFeatures.put(rankKey, example.example.get(0).floatFeatures.get(rankKey))
      val pos = new java.util.HashSet[String]()
      fv.stringFeatures.put("+", pos)

      val count = cand.size
      for (i <- 0 until count) {
        val resp = BoostedStumpsModel.getStumpResponse(cand(i), ex)
        if (resp) {
          pos.add(i.toString)
        }
      }
      output
    })
  }

  def trainAndSaveToFile(sc : SparkContext,
                         input : RDD[Example],
                         config : Config,
                         key : String) = {
    val model = train(sc, input, config, key)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }
}
