package com.airbnb.aerosolve.training

import java.util

import com.airbnb.aerosolve.core.{Example, ModelRecord}
import com.airbnb.aerosolve.core.features.{Family, FeatureRegistry, SimpleExample}
import com.airbnb.aerosolve.core.models.BoostedStumpsModel
import com.typesafe.config.Config
import org.apache.spark.SparkContext
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
            key : String,
            registry: FeatureRegistry) : BoostedStumpsModel = {
    val candidateSize : Int = config.getInt(key + ".num_candidates")
    val labelFamily : Family = registry.family(config.getString(key + ".rank_key"))

    val pointwise : RDD[Example] =
      LinearRankerUtils
        .makePointwiseFloat(input, config, key, registry)

    val candidates : Array[ModelRecord] = getCandidateStumps(pointwise, candidateSize, labelFamily)

    val data : RDD[Example] = getResponses(sc, pointwise, candidates, labelFamily).cache()

    val weights = LinearRankerTrainer.train(sc, data, config, key, registry).toMap

    // Lookup each candidate's weights
    (0 until candidates.size).foreach(i => {
      val stump = candidates(i)
      val pos = weights.getOrElse(registry.feature("+", i.toString), 0.0)
      stump.setFeatureWeight(pos)
    })

    val sorted = candidates.toBuffer.sortWith((a, b) =>
          math.abs(a.getFeatureWeight) > math.abs(b.getFeatureWeight))

    val stumps = new util.ArrayList[ModelRecord]()
    sorted.foreach(stump => {
      if (math.abs(stump.getFeatureWeight) > 0.0) {
        stumps.add(stump)
      }
    })

    val model = new BoostedStumpsModel(registry)
    model.stumps(stumps)
    model
  }

  def getCandidateStumps(pointwise : RDD[Example],
                         candidateSize : Int,
                         labelFamily : Family) : Array[ModelRecord] = {
    val result = collection.mutable.HashSet[ModelRecord]()
    pointwise
      .flatMap(example => example.only.iterator)
      .filter(featureValue => featureValue.feature.family != labelFamily)
      .take(candidateSize)
      .foreach(featureValue => {
        val rec = new ModelRecord()
        rec.setFeatureFamily(featureValue.feature.family.name)
        rec.setFeatureName(featureValue.feature.name)
        rec.setThreshold(featureValue.value)
        result.add(rec)
      })
      result.toArray
  }

  def getResponses(sc : SparkContext,
                   pointwise : RDD[Example],
                   candidates : Array[ModelRecord],
                   labelFamily : Family) : RDD[Example] = {
    val candidatesBC = sc.broadcast(candidates)
    pointwise.map(example => {
      val cand  = candidatesBC.value
      val ex = example.only
      val output = new SimpleExample(ex.registry)
      val fv = output.createVector()
      val labelFamilyVector = ex.get(labelFamily)
      labelFamilyVector.iterator.asScala.foreach(featureValue =>
        fv.put(featureValue.feature, featureValue.value)
      )
      val plusFamily = fv.registry.family("+")

      val count = cand.length
      for (i <- 0 until count) {
        val resp = BoostedStumpsModel.getStumpResponse(cand(i), ex)
        if (resp) {
          fv.putString(plusFamily.feature(i.toString))
        }
      }
      output
    })
  }

  def trainAndSaveToFile(sc : SparkContext,
                         input : RDD[Example],
                         config : Config,
                         key : String,
                         registry: FeatureRegistry) = {
    val model = train(sc, input, config, key, registry)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }
}
