package com.airbnb.aerosolve.training

import java.util

import com.airbnb.aerosolve.core.features.{Family, FeatureRegistry}
import com.airbnb.aerosolve.core.models.BoostedStumpsModel
import com.airbnb.aerosolve.core.models.DecisionTreeModel
import com.airbnb.aerosolve.core.models.ForestModel
import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.core.ModelRecord
import com.airbnb.aerosolve.core.util.Util
import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.util.Random
import scala.util.Try
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

// A tree forest trainer.
object ForestTrainer {
  private final val log: Logger = LoggerFactory.getLogger("ForestTrainer")

  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String,
            registry: FeatureRegistry) : ForestModel = {
    val candidateSize : Int = config.getInt(key + ".num_candidates")
    val labelFamily : Family = registry.family(config.getString(key + ".rank_key"))
    val rankThreshold : Double = config.getDouble(key + ".rank_threshold")
    val maxDepth : Int = config.getInt(key + ".max_depth")
    val minLeafCount : Int = config.getInt(key + ".min_leaf_items")
    val numTries : Int = config.getInt(key + ".num_tries")
    val splitCriteriaName : String = Try(config.getString(key + ".split_criteria"))
      .getOrElse("gini")
    
    val numTrees : Int = config.getInt(key + ".num_trees")

    val examples = LinearRankerUtils
        .makePointwiseFloat(input, config, key, registry)
        .map(example => example.only)
        .filter(vector => vector.contains(labelFamily))
        .coalesce(numTrees, true)
        
    val trees = examples.mapPartitions(part => {
      val ex = part
        .take(candidateSize)
        .toArray
      val stumps = new util.ArrayList[ModelRecord]()
      stumps.append(new ModelRecord)
      DecisionTreeTrainer.buildTree(
          stumps,
          ex,
          0,
          0,
          maxDepth,
          labelFamily,
          rankThreshold,
          numTries,
          minLeafCount,
          SplitCriteria.splitCriteriaFromName(splitCriteriaName))
      
      val tree = new DecisionTreeModel(registry)
      tree.stumps(stumps)
  
      Array(tree).iterator      
    })
    .collect
    
    log.info("%d trees trained".format(trees.size))
    
    val forest = new ForestModel(registry)
    val scale = 1.0f / numTrees.toFloat
    forest.trees(new java.util.ArrayList[DecisionTreeModel]())
    for (tree <- trees) {
      for (stump <- tree.stumps) {
        if (stump.getFeatureWeight != 0.0f) {
          stump.setFeatureWeight(stump.getFeatureWeight * scale)
        }
        if (stump.getLabelDistribution != null) {
          val dist = stump.getLabelDistribution.asScala
          for (rec <- dist) {
            dist.put(rec._1, rec._2 * scale)
          }
        }
      }
      forest.trees.append(tree)
    }
    forest
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
