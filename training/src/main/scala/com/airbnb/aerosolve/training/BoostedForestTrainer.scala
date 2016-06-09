package com.airbnb.aerosolve.training

import java.util

import com.airbnb.aerosolve.core.models.BoostedStumpsModel
import com.airbnb.aerosolve.core.models.DecisionTreeModel
import com.airbnb.aerosolve.core.models.ForestModel
import com.airbnb.aerosolve.core.{FeatureVector, Example, ModelRecord}
import com.airbnb.aerosolve.core.util.{FloatVector, Util}
import com.airbnb.aerosolve.training.GradientUtils.GradientContainer
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

  /**
    * Training parameters for boosted forest model:
    *
    * samplingStrategy: There are three sample strategies - "first", "uniform", "early_sample".
    * It is used to determine how to select samples for building candidate tree in each iteration of the training.
    * If "first" or "uniform" is selected, the trainer take "candidateSize" samples to build the tree;
    * "first" is much faster than "uniform" and "subsample_tree_candidates" should be either unspecified or 1.0.
    * If "early_sample" is selected, the trainer sample "subsample_tree_candidates" percentage of samples to build the tree
    * and in this case "candidateSize" is not used.
    *
    * subsample: percentage of training samples to be used for each iteration in the boosting step.
    *
    * max_depth, split_criteria, num_tries, min_leaf_items: parameters for building decision trees.
    *
    * multiclass: true if the model is for multi-class classification
    *
    * num_trees: number of trees in the forest
    *
    * iterations: number of iterations to run in each boosting step
    */
  case class BoostedForestTrainerParams(candidateSize: Int,
                                        rankKey: String,
                                        rankThreshold: Double,
                                        maxDepth: Int,
                                        minLeafCount: Int,
                                        numTries: Int,
                                        splitCriteria: String,
                                        numTrees: Int,
                                        subsample: Double,
                                        subsampleTreeCandidates: Double,
                                        iterations: Int,
                                        learningRate: Double,
                                        samplingStrategy: String,
                                        multiclass: Boolean,
                                        loss: String,
                                        margin: Double)

  // A container class that returns the tree and leaf a feature vector ends up in
  case class ForestResponse(tree : Int, leaf : Int)

  // The sum of all responses to a feature vector of an entire forest.
  case class ForestResult(label : Double,
                          labels : Map[String, Double],
                          sum : Double,
                          sumResponses : scala.collection.mutable.HashMap[String,Double],
                          response : Array[ForestResponse])

  private def loadTrainingParameters(taskConfig: Config): BoostedForestTrainerParams = {
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
    val loss : String = Try{taskConfig.getString("loss")}.getOrElse("logistic")
    val margin: Double = Try{taskConfig.getDouble("margin")}.getOrElse(1.0)
    // Note this is only used for 'early_sample' option
    val subsampleTreeCandidates: Double = Try{taskConfig.getDouble("subsample_tree_candidates")}.getOrElse(1.0)

    val params = BoostedForestTrainerParams(candidateSize = candidateSize,
      rankKey = rankKey,
      rankThreshold = rankThreshold,
      maxDepth = maxDepth,
      minLeafCount = minLeafCount,
      numTries = numTries,
      splitCriteria = splitCriteria,
      numTrees = numTrees,
      subsample = subsample,
      subsampleTreeCandidates = subsampleTreeCandidates,
      iterations = iterations,
      learningRate = learningRate,
      samplingStrategy = samplingStrategy,
      multiclass = splitCriteria.contains("multiclass"),
      loss = loss,
      margin = margin
    )
    params
  }

  def train(sc : SparkContext,
            input: Double => RDD[Example],
            config : Config,
            key : String) : ForestModel = {
    val taskConfig = config.getConfig(key)
    val params = loadTrainingParameters(taskConfig)
    val forest = new ForestModel()
    forest.setTrees(new java.util.ArrayList[DecisionTreeModel]())

    val transformed = (frac: Double) => LinearRankerUtils.makePointwiseFloat(input(frac), config, key)

    for (i <- 0 until params.numTrees) {
      log.info("Iteration %d".format(i))
      addNewTree(sc, forest, transformed, config, key, params)
      if (params.multiclass) {
        boostForestMulticlass(sc, forest, transformed, config, key, params)
      } else {
        boostForest(sc, forest, transformed, config, key, params)
      }
    }

    forest
  }
  
  def optionalExample(ex : Example, forest : ForestModel, params : BoostedForestTrainerParams)
  : Option[FeatureVector] = {
    val item = ex.example(0)
    if (params.multiclass) {
      val labels = TrainingUtils.getLabelDistribution(item, params.rankKey)
      if (labels.isEmpty) return None

      // Assuming that this is a single label multi-class
      val label = labels.head._1

      val scores = forest.scoreItemMulticlass(item)
      if (scores.isEmpty) return Some(item)

      forest.scoreToProbability(scores)

      val probs = scores.filter(x => x.label == label)
      if (probs.isEmpty) return None

      val importance = 1.0 - probs.head.probability
      if (scala.util.Random.nextDouble < importance) {
        Some(item)
      } else {
        None
      }
    } else {
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
  }

  def getSample(sc : SparkContext,
      forest: ForestModel,
      input : Double => RDD[Example],
      config : Config,
      key : String,
      params : BoostedForestTrainerParams) = {
    val forestBC = sc.broadcast(forest)
    val paramsBC = sc.broadcast(params)
    val examples = input(params.subsampleTreeCandidates)
              .map(x => optionalExample(x, forestBC.value, paramsBC.value))
              .filter(x => x.isDefined)
              .map(x => Util.flattenFeature(x.get))

     params.samplingStrategy match {
       // Picks the first few items that match the criteria. Better for massive data sets.
       case "first" => {
         examples.take(params.candidateSize)
       }
       // Picks uniformly. Better for small data sets.
       case "uniform" => examples.takeSample(false, params.candidateSize)
       case "early_sample" => examples.collect
     }
  }

  def addNewTree(sc : SparkContext,
      forest: ForestModel,
      input: Double => RDD[Example],
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
        params.rankKey,
        params.rankThreshold,
        Int.MaxValue,
        params.numTries,
        params.minLeafCount,
        SplitCriteria.splitCriteriaFromName(params.splitCriteria))
    
    val tree = new DecisionTreeModel()
    tree.setStumps(stumps)
    if (params.multiclass) {
      // Convert a pdf into something like a weight
      val scale = 1.0f / params.numTrees.toFloat
      for (stump <- tree.getStumps) {
        if (stump.labelDistribution != null) {
          val dist = stump.labelDistribution.asScala

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
      for (stump <- tree.getStumps) {
        if (stump.featureWeight != 0.0f) {
          stump.featureWeight *= scale
        }
      }
    }
    forest.getTrees.append(tree)
  }
  
  def boostForest(sc : SparkContext,
                  forest: ForestModel,
                  input: Double => RDD[Example],
                  config : Config,
                  key : String,
                  params : BoostedForestTrainerParams) = {
     for (i <- 0 until params.iterations) {
       log.info("Running boost iteration %d".format(i))
       val forestBC = sc.broadcast(forest)
       val paramsBC = sc.broadcast(params)
       // Get forest responses
       val labelSumResponse = input(params.subsample)
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
       val trees = forest.getTrees.asScala.toArray
       var sum : Double = 0.0
       for (cg <- countAndGradient) {
         val key = cg._1
         val tree = trees(key._1)
         val stump = tree.getStumps.get(key._2)
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
                  input : Double => RDD[Example],
                  config : Config,
                  key : String,
                  params : BoostedForestTrainerParams) = {
    for (i <- 0 until params.iterations) {
      log.info("Running boost iteration %d".format(i))
      val forestBC = sc.broadcast(forest)
      // Get forest responses
      val labelSumResponse = input(params.subsample)
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
      val trees = forest.getTrees.asScala.toArray
      var sum : Double = 0.0
      for (cg <- countAndGradient) {
        val key = cg._1
        val tree = trees(key._1)
        val stump = tree.getStumps.get(key._2)
        val label = key._3
        val (count, grad) = cg._2
        val curr = stump.labelDistribution.get(label)
        val avgGrad = grad / count
        if (curr == null) {
          stump.labelDistribution.put(label, - params.learningRate * avgGrad)
        } else {
          stump.labelDistribution.put(label, curr - params.learningRate * avgGrad)
        }
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
    val sumResponses = scala.collection.mutable.HashMap[String, Double]()
    for (i <- 0 until trees.size) {
      val tree = trees(i)
      val leaf = trees(i).getLeafIndex(floatFeatures)
      if (leaf >= 0) {
        val stump = tree.getStumps().get(leaf)
        val weight = stump.featureWeight
        sum = sum + weight
        if (params.multiclass && stump.labelDistribution != null) {
          val dist = stump.labelDistribution.asScala
          for (kv <- dist) {
            val v = sumResponses.getOrElse(kv._1, 0.0)
            sumResponses.put(kv._1, v + kv._2)
          }
        }
        val response = ForestResponse(i, leaf)
        result.append(response)
      }      
    }
    val label = TrainingUtils.getLabel(item, params.rankKey, params.rankThreshold)
    val labels = TrainingUtils.getLabelDistribution(item, params.rankKey)
    ForestResult(label, labels, sum, sumResponses, result.toArray)
  }
  
  def trainAndSaveToFile(sc : SparkContext,
                         input : RDD[Example],
                         config : Config,
                         key : String) = {
   trainAndSaveToFileEarlySample(sc, (frac: Double) => input.sample(false, frac), config, key)
  }

  /**
    * Entry point to train and persist model on disk
    *
    * This version allows sample to be pushed down in the Example loading process.
    * One use case is to avoid deserialization when examples are discarded by sampling.
    *
    * @note Care should be taken when caching dataset as the order of cache and sample call will determine the proportion
    * of dataset be cached and whether each reference will result in a new set of sample.
    * @param sampleInput a function takes sampling fraction and returns sampled dataset
    */
  def trainAndSaveToFileEarlySample(sc: SparkContext,
                                    sampleInput: Double => RDD[Example],
                                    config: Config,
                                    key: String) = {
    val model = train(sc, sampleInput, config, key)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }
}
