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

// A boosted tree forest trainer.
// Alternates between fitting a tree and building one on the importance
// sampled outliers of previous trees.
object BoostedForestTrainer {
  private final val log: Logger = LoggerFactory.getLogger("BoostedForestTrainer")
  
  case class BoostedForestTrainerOptions(candidateSize : Int,
                                         rankKey : String,
                                         rankThreshold : Double,
                                         maxDepth : Int,
                                         minLeafCount : Int,
                                         numTries : Int,
                                         splitCriteria : String,
                                         numTrees :Int,
                                         subsample : Double,
                                         iterations : Int,
                                         learningRate : Double)
                                         
  case class ForestResponse(tree : Int, leaf : Int, weight : Double)
  case class ForestResult(label : Double, sum : Double, response : Array[ForestResponse])

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
    val splitCriteria : String = Try(config.getString(key + ".split_criteria")).getOrElse("gini")
    
    val iterations : Int = config.getInt(key + ".iterations")
    val subsample : Double = config.getDouble(key + ".subsample")
    val learningRate : Double = config.getDouble(key + ".learning_rate")
    val numTrees : Int = config.getInt(key + ".num_trees")
    
    val opt = BoostedForestTrainerOptions(candidateSize = candidateSize,
          rankKey = rankKey,
          rankThreshold = rankThreshold,
          maxDepth = maxDepth,
          minLeafCount = minLeafCount,
          numTries = numTries,
          splitCriteria = splitCriteria,
          numTrees = numTrees,
          subsample = subsample,
          iterations = iterations,
          learningRate = learningRate
        )
    
    val forest = new ForestModel()
    forest.setTrees(new java.util.ArrayList[DecisionTreeModel]())
    
    for (i <- 0 until numTrees) {
      log.info("Iteration %d".format(i))
      addNewTree(sc, forest, input, config, key, opt)
      boostForest(sc, forest, input, config, key, opt)
    }
    
    forest
  }
  
  def addNewTree(sc : SparkContext,
      forest: ForestModel,
      input : RDD[Example],
      config : Config,
      key : String,
      opt : BoostedForestTrainerOptions) = {
    val forestBC = sc.broadcast(forest)
    val ex = LinearRankerUtils
              .makePointwiseFloat(input, config, key)
              .map(x => {
                val item = x.example(0)
                val label = TrainingUtils.getLabel(item, opt.rankKey, opt.rankThreshold)
                val localForest = forestBC.value
                val score = localForest.scoreItem(x.example(0))
                val prob = localForest.scoreProbability(score)
                val importance = 1.0 - label * prob
                if (scala.util.Random.nextDouble < importance) {
                  Some(item)
                } else {
                  None
                }
              })
              .filter(x => x != None)
              .map(x => Util.flattenFeature(x.get))
              .take(opt.candidateSize)
              .toArray
    val stumps = new util.ArrayList[ModelRecord]()
    stumps.append(new ModelRecord)
    DecisionTreeTrainer.buildTree(
        stumps,
        ex,
        0,
        0,
        opt.maxDepth,
        opt.rankKey,
        opt.rankThreshold,
        opt.numTries,
        opt.minLeafCount,
        opt.splitCriteria)
    
    val tree = new DecisionTreeModel()
    tree.setStumps(stumps)
    val scale = 1.0f / opt.numTrees.toFloat
    for (stump <- tree.getStumps) {
      if (stump.featureWeight != 0.0f) {
        stump.featureWeight *= scale
      }
    }
    forest.getTrees().append(tree)
  }
  
  def boostForest(sc : SparkContext,
                  forest: ForestModel,
                  input : RDD[Example],
                  config : Config,
                  key : String,
                  opt : BoostedForestTrainerOptions) = {
     for (i <- 0 until opt.iterations) {
       log.info("Running boost iteration %d".format(i))
       val forestBC = sc.broadcast(forest)
       // Get forest responses
       val labelSumResponse = LinearRankerUtils
              .makePointwiseFloat(input, config, key)
              .sample(false, opt.subsample)
              .map(x => getForestResponse(forestBC.value, x, opt))
       // Get forest batch gradient
       val countAndGradient = labelSumResponse.mapPartitions(part => {
         val gradientMap = scala.collection.mutable.HashMap[(Int, Int), (Double, Double)]()
         part.foreach(x => {
           val corr = scala.math.min(10.0, x.label * x.sum)
           val expCorr = scala.math.exp(corr)
           val grad = -x.label / (1.0 + expCorr)
           x.response.foreach(resp => {
             val key = (resp.tree, resp.leaf)
             val current = gradientMap.getOrElse(key, (0.0, 0.0))
             gradientMap.put(key, (current._1 + 1, current._2 + grad))
           })         
         })
         gradientMap.iterator
       })
       .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
       .collectAsMap
       
       // Gradient step
       val trees = forest.getTrees().asScala.toArray
       var sum : Double = 0.0
       for (cg <- countAndGradient) {
         val key = cg._1
         val tree = trees(key._1)
         val stump = tree.getStumps().get(key._2)
         val (count, grad) = cg._2
         val curr = stump.getFeatureWeight()
         val avgGrad = grad / count
         stump.setFeatureWeight(curr - opt.learningRate * avgGrad)
         sum = avgGrad * avgGrad
       }
       log.info("Gradient L2 Norm = %f".format(scala.math.sqrt(sum)))
     }
  }
  
  def getForestResponse(forest : ForestModel,
                        ex : Example,
                        opt : BoostedForestTrainerOptions) : ForestResult = {
    val item = ex.example.get(0)
    val floatFeatures = Util.flattenFeature(item)
    val result = scala.collection.mutable.ArrayBuffer[ForestResponse]()
    val trees = forest.getTrees().asScala.toArray
    var sum : Double = 0.0
    for (i <- 0 until trees.size) {
      val tree = trees(i)
      val leaf = trees(i).getLeafIndex(floatFeatures);
      if (leaf >= 0) {
        val stump = tree.getStumps().get(leaf);
        val weight = stump.featureWeight
        sum = sum + weight
        val response = ForestResponse(i, leaf, sum) 
        result.append(response)
      }      
    }
    val label = TrainingUtils.getLabel(item, opt.rankKey, opt.rankThreshold)
    ForestResult(label, sum, result.toArray)
  }
  
  def trainAndSaveToFile(sc : SparkContext,
                         input : RDD[Example],
                         config : Config,
                         key : String) = {
    val model = train(sc, input, config, key)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }
}
