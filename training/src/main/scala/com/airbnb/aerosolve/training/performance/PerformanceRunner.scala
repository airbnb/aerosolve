package com.airbnb.aerosolve.training.performance

import java.util

import com.airbnb.aerosolve.core.models.DecisionTreeModel
import com.airbnb.aerosolve.core.transforms.Transformer
import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.core.{Example, FeatureVector, ModelRecord}
import com.airbnb.aerosolve.training.{DecisionTreeTrainer, LinearRankerUtils, SplitCriteria}
import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.util.{Random, Try}

object PerformanceRunner {
  def makeConfig(splitCriteria : String) : String = {
    """
      |identity_transform {
      |  transform : list
      |  transforms: []
      |}
      |model_config {
      |  rank_key : "$rank"
      |  split_criteria : "%s"
      |  num_candidates : 5000
      |  rank_threshold : 0.0
      |  max_depth : 10
      |  min_leaf_items : 5
      |  num_tries : 100
      |  context_transform : identity_transform
      |  item_transform : identity_transform
      |  combined_transform : identity_transform
      |}
    """.stripMargin
      .format(splitCriteria)
  }

  def generateData(numExamples: Int): Array[Example] = {
    (0 to numExamples).map(index => {
      val example = new Example
      val item: FeatureVector = new FeatureVector

      item.setFloatFeatures(new java.util.HashMap)
      item.setStringFeatures(new java.util.HashMap)

      val floatFeatures = item.getFloatFeatures
      val stringFeatures = item.getStringFeatures

      floatFeatures.put("$rank", new java.util.HashMap)
      floatFeatures.get("$rank").put("", Random.nextDouble() * 100.0)

      floatFeatures.put("loc", new java.util.HashMap)

      val loc = floatFeatures.get("loc")

      loc.put("v", Random.nextDouble())
      loc.put("w", Random.nextDouble())
      loc.put("x", Random.nextDouble())
      loc.put("y", Random.nextDouble())
      loc.put("z", Random.nextDouble())

      example.addToExample(item)

      example
    }).toArray
  }

  // Makes an example pointwise while preserving the float features.
  def makePointwiseFloatLocal(
      examples : Array[Example],
      config : Config,
      key : String) : Array[Example] = {
    val transformer = new Transformer(config, key)

    examples.map(example => {
      val buffer = collection.mutable.ArrayBuffer[Example]()
      example.example.asScala.foreach(x => {
        val newExample = new Example()
        newExample.setContext(example.context)
        newExample.addToExample(x)
        transformer.combineContextAndItems(newExample)
        buffer.append(newExample)
      })
      buffer
    })
    .flatMap(x => x)
  }

  def trainLocal(
      input: Array[Example],
      config: Config,
      key: String) : DecisionTreeModel = {
    val candidateSize : Int = config.getInt(key + ".num_candidates")
    val rankKey : String = config.getString(key + ".rank_key")
    val rankThreshold : Double = config.getDouble(key + ".rank_threshold")
    val maxDepth : Int = config.getInt(key + ".max_depth")
    val minLeafCount : Int = config.getInt(key + ".min_leaf_items")
    val numTries : Int = config.getInt(key + ".num_tries")
    val splitCriteriaName : String = Try(config.getString(key + ".split_criteria"))
      .getOrElse("gini")

    val examples = makePointwiseFloatLocal(input, config, key)
      .map(x => Util.flattenFeature(x.example(0)))
      .filter(x => x.contains(rankKey))
      .take(candidateSize)

    val stumps = new util.ArrayList[ModelRecord]()
    stumps.append(new ModelRecord)

    DecisionTreeTrainer.buildTree(
      stumps,
      examples,
      0,
      0,
      maxDepth,
      rankKey,
      rankThreshold,
      numTries,
      minLeafCount,
      SplitCriteria.splitCriteriaFromName(splitCriteriaName)
    )

    val model = new DecisionTreeModel()
    model.setStumps(stumps)

    model
  }


  def main(args: Array[String]): Unit = {
    val config = ConfigFactory.parseString(makeConfig("variance"))

    (0 to 1000).foreach(index => {
      println("Iteration: %d".format(index))

      val examples = generateData(10000)

      val model = trainLocal(examples, config, "model_config")
    })

    println("Done!")
  }
}
