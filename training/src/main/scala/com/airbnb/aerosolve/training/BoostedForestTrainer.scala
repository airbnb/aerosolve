package com.airbnb.aerosolve.training

import java.util

import com.airbnb.aerosolve.core.features.{Family, FeatureRegistry, MultiFamilyVector}
import com.airbnb.aerosolve.core.models.{DecisionTreeModel, ForestModel}
import com.airbnb.aerosolve.core.{Example, ModelRecord}
import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.util.Try

// A boosted tree forest trainer.
// Alternates between fitting a tree and building one on the importance
// sampled outliers of previous trees.
// https://en.wikipedia.org/wiki/Gradient_boosting
object BoostedForestTrainer {
  private final val log: Logger = LoggerFactory.getLogger("BoostedForestTrainer")
  
  case class BoostedForestTrainerParams(candidateSize : Int,
                                         labelFamily : Family,
                                         rankThreshold : Double,
                                         maxDepth : Int,
                                         minLeafCount : Int,
                                         numTries : Int,
                                         splitCriteria : String,
                                         numTrees :Int,
                                         subsample : Double,
                                         iterations : Int,
                                         learningRate : Double,
                                         samplingStrategy : String,
                                         multiclass : Boolean,
                                         loss : String,
                                         margin : Double,
                                         registry: FeatureRegistry)
  // A container class that returns the tree and leaf a feature vector ends up in
  case class ForestResponse(tree : Int, leaf : Int)
  // The sum of all responses to a feature vector of an entire forest.
  case class ForestResult(label : Double,
                          labels : Map[String, Double],
                          sum : Double,
                          sumResponses : scala.collection.mutable.HashMap[String,Double],
                          response : Array[ForestResponse])

  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String,
            registry: FeatureRegistry) : ForestModel = {
    val taskConfig = config.getConfig(key)
    val candidateSize : Int = taskConfig.getInt("num_candidates")
    val labelFamily : Family = registry.family(taskConfig.getString("rank_key"))
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
    val loss : String = Try{taskConfig.getString("loss")}.getOrElse("logistic")
    val margin: Double = Try{taskConfig.getDouble("margin")}.getOrElse(1.0)
    val cache: String = Try{taskConfig.getString("cache")}.getOrElse("")

    val params = BoostedForestTrainerParams(candidateSize = candidateSize,
          labelFamily = labelFamily,
          rankThreshold = rankThreshold,
          maxDepth = maxDepth,
          minLeafCount = minLeafCount,
          numTries = numTries,
          splitCriteria = splitCriteria,
          numTrees = numTrees,
          subsample = subsample,
          iterations = iterations,
          learningRate = learningRate,
          samplingStrategy = samplingStrategy,
          multiclass = splitCriteria.contains("multiclass"),
          loss = loss,
          margin = margin,
          registry = registry
        )
    
    val forest = new ForestModel(registry)
    forest.trees(new java.util.ArrayList[DecisionTreeModel]())

    val raw = LinearRankerUtils.makePointwiseFloat(input, config, key, registry)

    val examples = cache match {
      case "memory" => raw.cache()
      case _ : String => raw
    }

    for (i <- 0 until numTrees) {
      log.info("Iteration %d".format(i))
      addNewTree(sc, forest, input, config, key, params)
      if (params.multiclass) {
        boostForestMulticlass(sc, forest, examples, config, key, params)
      } else {
        boostForest(sc, forest, examples, config, key, params)
      }
    }

    cache match {
      case "memory" => examples.unpersist()
      case _ : String =>
    }
    
    forest
  }
  
  def optionalExample(ex : Example, forest : ForestModel, params : BoostedForestTrainerParams)
  : Option[MultiFamilyVector] = {
    val item = ex.only
    if (params.multiclass) {
      val labels = TrainingUtils.getLabelDistribution(item, params.labelFamily)
      if (labels.isEmpty) return None

      // Assuming that this is a single label multi-class
      val label = labels.head._1

      val scores = forest.scoreItemMulticlass(item)
      if (scores.isEmpty) return Some(item)

      forest.scoreToProbability(scores)

      val probs = scores.filter(x => x.getLabel == label.name())
      if (probs.isEmpty) return None

      val importance = 1.0 - probs.head.getProbability
      if (scala.util.Random.nextDouble < importance) {
        Some(item)
      } else {
        None
      }
    } else {
      val label = TrainingUtils.getLabel(item, params.labelFamily, params.rankThreshold)
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
  }

  def getSample(sc : SparkContext,
      forest: ForestModel,
      input : RDD[Example],
      config : Config,
      key : String,
      params : BoostedForestTrainerParams) = {
    val forestBC = sc.broadcast(forest)
    val paramsBC = sc.broadcast(params)
    val examples = input
              .flatMap(x => optionalExample(x, forestBC.value, paramsBC.value))

     params.samplingStrategy match {
       // Picks the first few items that match the criteria. Better for massive data sets.
       case "first" => examples.take(params.candidateSize)
       // Picks uniformly. Better for small data sets.
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
    log.info("Got %d examples".format(ex.length))
    val stumps = new util.ArrayList[ModelRecord]()
    stumps.append(new ModelRecord)
    DecisionTreeTrainer.buildTree(
        stumps,
        ex,
        0,
        0,
        params.maxDepth,
        params.labelFamily,
        params.rankThreshold,
        params.numTries,
        params.minLeafCount,
        SplitCriteria.splitCriteriaFromName(params.splitCriteria))
    
    val tree = new DecisionTreeModel(params.registry)
    tree.stumps(stumps)
    if (params.multiclass) {
      // Convert a pdf into something like a weight
      val scale = 1.0f / params.numTrees.toFloat
      for (stump <- tree.stumps) {
        if (stump.getLabelDistribution != null) {
          val dist = stump.getLabelDistribution.asScala

          for (key <- dist.keys) {
            val v = dist.get(key)
            if (v.isDefined) {
              dist.put(key, scale * (2.0 * v.get - 1.0))
            }
          }
        }
      }
    } else {
      val scale = 1.0f / params.numTrees.toFloat
      for (stump <- tree.stumps) {
        if (stump.getFeatureWeight != 0.0f) {
          stump.setFeatureWeight(stump.getFeatureWeight * scale)
        }
      }
    }
    forest.trees.append(tree)
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
       val paramsBC = sc.broadcast(params)
       // Get forest responses
       val labelSumResponse = input
              .sample(false, params.subsample)
              .map(x => getForestResponse(forestBC.value, x, params))
       // Get forest batch gradient
       val countAndGradient = labelSumResponse.mapPartitions(part => {
         val gradientMap = scala.collection.mutable.HashMap[(Int, Int), (Double, Double)]()
         part.foreach(x => {
           val grad = paramsBC.value.loss match {
             case "hinge" => {
               val loss = scala.math.max(0.0, paramsBC.value.margin - x.label * x.sum)
               if (loss > 0.0) -x.label else 0.0
             }
             case _ => {
               val corr = scala.math.min(10.0, x.label * x.sum)
               val expCorr = scala.math.exp(corr)
               -x.label / (1.0 + expCorr)
             }
           }
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
       val trees = forest.trees.asScala.toArray
       var sum : Double = 0.0
       for (cg <- countAndGradient) {
         val key = cg._1
         val tree = trees(key._1)
         val stump = tree.stumps.get(key._2)
         val (count, grad) = cg._2
         val curr = stump.getFeatureWeight
         val avgGrad = grad / count
         stump.setFeatureWeight(curr - params.learningRate * avgGrad)
         sum += avgGrad * avgGrad
       }
       log.info("Gradient L2 Norm = %f".format(scala.math.sqrt(sum)))
     }
  }

  def boostForestMulticlass(
                  sc : SparkContext,
                  forest: ForestModel,
                  input : RDD[Example],
                  config : Config,
                  key : String,
                  params : BoostedForestTrainerParams) = {
    for (i <- 0 until params.iterations) {
      log.info("Running boost iteration %d".format(i))
      val forestBC = sc.broadcast(forest)
      // Get forest responses
      val labelSumResponse = input
        .sample(false, params.subsample)
        .map(x => getForestResponse(forestBC.value, x, params))
      // Get forest batch gradient
      val countAndGradient = labelSumResponse.mapPartitions(part => {
        val gradientMap = scala.collection.mutable.HashMap[(Int, Int, String), (Double, Double)]()
        part.foreach(x => {
          val posLabels = x.labels.keys
          // Convert to multinomial using softmax.
          val max = x.sumResponses.values.max
          val tmp = x.sumResponses.map(x => (x._1, Math.exp(x._2 - max)))
          val sum = math.max(1e-10, tmp.values.sum)
          val importance = tmp
            .map(x => (x._1, x._2 / sum))
            .map(x => (x._1, if (posLabels.contains(x._1)) x._2 - 1.0 else x._2))

          x.response.foreach(resp => {
            importance.foreach(imp => {
              val key = (resp.tree, resp.leaf, imp._1)
              val current = gradientMap.getOrElse(key, (0.0, 0.0))
              val grad = imp._2
              gradientMap.put(key, (current._1 + 1, current._2 + grad))
            })
          })
        })
        gradientMap.iterator
      })
        .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
        .collectAsMap

      // Gradient step
      val trees = forest.trees.asScala.toArray
      var sum : Double = 0.0
      for (cg <- countAndGradient) {
        val key = cg._1
        val tree = trees(key._1)
        val stump = tree.stumps.get(key._2)
        val label = key._3
        val (count, grad) = cg._2
        val curr = stump.getLabelDistribution.get(label)
        val avgGrad = grad / count
        if (curr == null) {
          stump.getLabelDistribution.put(label, - params.learningRate * avgGrad)
        } else {
          stump.getLabelDistribution.put(label, curr - params.learningRate * avgGrad)
        }
        sum += avgGrad * avgGrad
      }
      log.info("Gradient L2 Norm = %f".format(scala.math.sqrt(sum)))
    }
  }
  
  def getForestResponse(forest : ForestModel,
                        ex : Example,
                        params : BoostedForestTrainerParams) : ForestResult = {
    val item = ex.only
    val result = scala.collection.mutable.ArrayBuffer[ForestResponse]()
    val trees = forest.trees.asScala.toArray
    var sum : Double = 0.0
    val sumResponses = scala.collection.mutable.HashMap[String, Double]()
    for (i <- trees.indices) {
      val tree = trees(i)
      val leaf = trees(i).getLeafIndex(item)
      if (leaf >= 0) {
        val stump = tree.stumps.get(leaf);
        val weight = stump.getFeatureWeight
        sum = sum + weight
        if (params.multiclass && stump.getLabelDistribution != null) {
          val dist = stump.getLabelDistribution.asScala
          for (kv <- dist) {
            val v = sumResponses.getOrElse(kv._1, 0.0)
            sumResponses.put(kv._1, v + kv._2)
          }
        }
        val response = ForestResponse(i, leaf)
        result.append(response)
      }      
    }
    val label = TrainingUtils.getLabel(item, params.labelFamily, params.rankThreshold)
    val labels: Map[String, Double] = TrainingUtils.getLabelDistribution(item, params.labelFamily)
      .map(kv => (kv._1.name, kv._2))
      .toMap
    ForestResult(label, labels, sum, sumResponses, result.toArray)
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
