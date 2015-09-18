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
// https://en.wikipedia.org/wiki/Gradient_boosting
object BoostedForestTrainer {
  private final val log: Logger = LoggerFactory.getLogger("BoostedForestTrainer")
  
  case class BoostedForestTrainerParams(candidateSize : Int,
                                         rankKey : String,
                                         rankThreshold : Double,
                                         maxDepth : Int,
                                         minLeafCount : Int,
                                         numTries : Int,
                                         splitCriteria : String,
                                         numTrees :Int,
                                         subsample : Double,
                                         iterations : Int,
                                         learningRate : Double,
                                         samplingStrategy : String)
                                         
  case class ForestResponse(tree : Int, leaf : Int, weight : Double)
  case class ForestResult(label : Double, sum : Double, response : Array[ForestResponse])

  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String) : ForestModel = {
    val taskConfig = config.getConfig(key)
    val candidateSize : Int = taskConfig.getInt("num_candidates")
    val rankKey : String = taskConfig.getString("rank_key")
    val rankThreshold : Double = taskConfig.getDouble("rank_threshold")
    val maxDepth : Int = taskConfig.getInt("max_depth")
    val minLeafCount : Int = taskConfig.getInt("min_leaf_items")
    val numTries : Int = taskConfig.getInt("num_tries")
    val splitCriteria : String = Try(taskConfig.getString("split_criteria")).getOrElse("gini")
    
    val iterations : Int = taskConfig.getInt("iterations")
    val subsample : Double = taskConfig.getDouble("subsample")
    val learningRate : Double = taskConfig.getDouble("learning_rate")
    val numTrees : Int = taskConfig.getInt("num_trees")
    val samplingStrategy : String = taskConfig.getString("sampling_strategy")
    
    val params = BoostedForestTrainerParams(candidateSize = candidateSize,
          rankKey = rankKey,
          rankThreshold = rankThreshold,
          maxDepth = maxDepth,
          minLeafCount = minLeafCount,
          numTries = numTries,
          splitCriteria = splitCriteria,
          numTrees = numTrees,
          subsample = subsample,
          iterations = iterations,
          learningRate = learningRate,
          samplingStrategy = samplingStrategy
        )
    
    val forest = new ForestModel()
    forest.setTrees(new java.util.ArrayList[DecisionTreeModel]())
    
    for (i <- 0 until numTrees) {
      log.info("Iteration %d".format(i))
      addNewTree(sc, forest, input, config, key, params)
      boostForest(sc, forest, input, config, key, params)
    }
    
    forest
  }
  
  def optionalExample(ex : Example, forest : ForestModel, params : BoostedForestTrainerParams) = {
    val item = ex.example(0)
    val label = TrainingUtils.getLabel(item, params.rankKey, params.rankThreshold)
    val score = forest.scoreItem(item)
    val prob = forest.scoreProbability(score)
    val importance = if (label > 0) {
      1.0 - prob
    } else {
      prob
    }
    if (scala.util.Random.nextDouble < importance) {
      Some(item)
    } else {
      None
    }
  }

  def getSample(sc : SparkContext,
      forest: ForestModel,
      input : RDD[Example],
      config : Config,
      key : String,
      params : BoostedForestTrainerParams) = {
    val forestBC = sc.broadcast(forest)
    val paramsBC = sc.broadcast(params)
    val examples = LinearRankerUtils
              .makePointwiseFloat(input, config, key)
              .map(x => optionalExample(x, forestBC.value, paramsBC.value))
              .filter(x => x != None)
              .map(x => Util.flattenFeature(x.get))

     params.samplingStrategy match {
       // Picks the first few items that match the criteria. Better for massive data sets.
       case "first" => {
         examples.take(params.candidateSize)
       }
       // Picks uniformly. Beter for small data sets.
       case "uniform" => examples.takeSample(false, params.candidateSize)
     }
  }

  def addNewTree(sc : SparkContext,
      forest: ForestModel,
      input : RDD[Example],
      config : Config,
      key : String,
      params : BoostedForestTrainerParams) = {
    val ex = getSample(sc, forest, input, config, key, params)
    log.info("Got %d examples".format(ex.size))
    val stumps = new util.ArrayList[ModelRecord]()
    stumps.append(new ModelRecord)
    DecisionTreeTrainer.buildTree(
        stumps,
        ex,
        0,
        0,
        params.maxDepth,
        params.rankKey,
        params.rankThreshold,
        params.numTries,
        params.minLeafCount,
        params.splitCriteria)
    
    val tree = new DecisionTreeModel()
    tree.setStumps(stumps)
    val scale = 1.0f / params.numTrees.toFloat
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
                  params : BoostedForestTrainerParams) = {
     for (i <- 0 until params.iterations) {
       log.info("Running boost iteration %d".format(i))
       val forestBC = sc.broadcast(forest)
       // Get forest responses
       val labelSumResponse = LinearRankerUtils
              .makePointwiseFloat(input, config, key)
              .sample(false, params.subsample)
              .map(x => getForestResponse(forestBC.value, x, params))
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
         stump.setFeatureWeight(curr - params.learningRate * avgGrad)
         sum += avgGrad * avgGrad
       }
       log.info("Gradient L2 Norm = %f".format(scala.math.sqrt(sum)))
     }
  }
  
  def getForestResponse(forest : ForestModel,
                        ex : Example,
                        params : BoostedForestTrainerParams) : ForestResult = {
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
    val label = TrainingUtils.getLabel(item, params.rankKey, params.rankThreshold)
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
