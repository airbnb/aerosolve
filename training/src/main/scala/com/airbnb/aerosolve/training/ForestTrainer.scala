package com.airbnb.aerosolve.training

import java.util

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
            key : String) : ForestModel = {
    val candidateSize : Int = config.getInt(key + ".num_candidates")
    val rankKey : String = config.getString(key + ".rank_key")
    val rankThreshold : Double = config.getDouble(key + ".rank_threshold")
    val maxDepth : Int = config.getInt(key + ".max_depth")
    val minLeafCount : Int = config.getInt(key + ".min_leaf_items")
    val numTries : Int = config.getInt(key + ".num_tries")
    val splitCriteriaName : String = Try(config.getString(key + ".split_criteria"))
      .getOrElse("gini")
    
    val numTrees : Int = config.getInt(key + ".num_trees")

    val examples = LinearRankerUtils
        .makePointwiseFloat(input, config, key)
        .map(x => Util.flattenFeature(x.example(0)))
        .filter(x => x.contains(rankKey))
        .coalesce(numTrees, true)

    val ex = examples.take(1)(0)
    val numFeatures = LinearRankerUtils.getNumFeatures(ex, rankKey)
    val maxFeatures : Int = Try(config.getString(key + ".max_features")).getOrElse("sqrt") match {
      case "all" => Int.MaxValue
      case "sqrt" => math.sqrt(numFeatures).ceil.toInt
      case "log2" => math.max(1, (math.log(numFeatures) / math.log(2)).ceil.toInt)
      case _ => Int.MaxValue
    }
        
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
          rankKey,
          rankThreshold,
          maxFeatures,
          numTries,
          minLeafCount,
          SplitCriteria.splitCriteriaFromName(splitCriteriaName))
      
      val tree = new DecisionTreeModel()
      tree.setStumps(stumps)
  
      Array(tree).iterator      
    })
    .collect
    .toArray
    
    log.info("%d trees trained".format(trees.size))
    
    val forest = new ForestModel()
    val scale = 1.0f / numTrees.toFloat
    forest.setTrees(new java.util.ArrayList[DecisionTreeModel]())
    for (tree <- trees) {
      for (stump <- tree.getStumps) {
        if (stump.featureWeight != 0.0f) {
          stump.featureWeight *= scale
        }
        if (stump.labelDistribution != null) {
          val dist = stump.labelDistribution.asScala
          for (rec <- dist) {
            dist.put(rec._1, rec._2 * scale)
          }
        }
      }
      forest.getTrees().append(tree)
    }
    forest
  }
  
  def trainAndSaveToFile(sc : SparkContext,
                         input : RDD[Example],
                         config : Config,
                         key : String) = {
    val model = train(sc, input, config, key)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }
}
