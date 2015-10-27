package com.airbnb.aerosolve.training

import java.util

import com.airbnb.aerosolve.core.models.BoostedStumpsModel
import com.airbnb.aerosolve.core.models.DecisionTreeModel
import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.core.ModelRecord
import com.airbnb.aerosolve.core.util.Util
import com.typesafe.config.Config
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

import scala.util.Random
import scala.util.Try
import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

// Types of split criteria
object SplitCriteriaTypes extends Enumeration {
  val Classification, Regression = Value
}

// The decision tree is meant to be a prior for the spline model / linear model
object DecisionTreeTrainer {
  private final val log: Logger = LoggerFactory.getLogger("DecisionTreeTrainer")

  // Mapping from each valid splitCriterion to its type
  private final val SplitCriteria = Map(
    "gini" -> SplitCriteriaTypes.Classification,
    "information_gain" -> SplitCriteriaTypes.Classification,
    "hellinger" -> SplitCriteriaTypes.Classification,
    "variance" -> SplitCriteriaTypes.Regression
  )

  def train(sc : SparkContext,
            input : RDD[Example],
            config : Config,
            key : String) : DecisionTreeModel = {
    val candidateSize : Int = config.getInt(key + ".num_candidates")
    val rankKey : String = config.getString(key + ".rank_key")
    val rankThreshold : Double = config.getDouble(key + ".rank_threshold")
    val maxDepth : Int = config.getInt(key + ".max_depth")
    val minLeafCount : Int = config.getInt(key + ".min_leaf_items")
    val numTries : Int = config.getInt(key + ".num_tries")
    val splitCriteria : String = Try(config.getString(key + ".split_criteria")).getOrElse("gini")

    val examples = LinearRankerUtils
        .makePointwiseFloat(input, config, key)
        .map(x => Util.flattenFeature(x.example(0)))
        .filter(x => x.contains(rankKey))
        .take(candidateSize)
    
    val stumps = new util.ArrayList[ModelRecord]()
    stumps.append(new ModelRecord)
    buildTree(stumps, examples, 0, 0, maxDepth, rankKey, rankThreshold, numTries, minLeafCount, splitCriteria)
    
    val model = new DecisionTreeModel()
    model.setStumps(stumps)

    model
  }
  
  def buildTree(stumps : util.ArrayList[ModelRecord],
                examples :  Array[util.Map[java.lang.String, util.Map[java.lang.String, java.lang.Double]]],
                currIdx : Int,
                currDepth : Int,
                maxDepth : Int,
                rankKey : String,
                rankThreshold : Double,
                numTries : Int,
                minLeafCount : Int,
                splitCriteria : String) : Unit = {
    if (currDepth >= maxDepth) {
      stumps(currIdx) = makeLeaf(examples, rankKey, rankThreshold, splitCriteria)
      return
    }
    val split = getBestSplit(examples, rankKey, rankThreshold, numTries, minLeafCount, splitCriteria)

    if (split == None) {
      stumps(currIdx) = makeLeaf(examples, rankKey, rankThreshold, splitCriteria)
      return
    }

    // This is a split node.
    stumps(currIdx) = split.get    
    val left = stumps.size
    stumps.append(new ModelRecord())
    val right = stumps.size
    stumps.append(new ModelRecord())
    stumps(currIdx).setLeftChild(left)
    stumps(currIdx).setRightChild(right)
    
    buildTree(stumps,
              examples.filter(x => BoostedStumpsModel.getStumpResponse(stumps(currIdx), x) == false),
              left,
              currDepth + 1,
              maxDepth,
              rankKey,
              rankThreshold,
              numTries,
              minLeafCount,
              splitCriteria)
    buildTree(stumps,
              examples.filter(x => BoostedStumpsModel.getStumpResponse(stumps(currIdx), x) == true),
              right,
              currDepth + 1,
              maxDepth,
              rankKey,
              rankThreshold,
              numTries,
              minLeafCount,
              splitCriteria)    
  }
  
  def makeLeaf(examples :  Array[util.Map[java.lang.String, util.Map[java.lang.String, java.lang.Double]]],
               rankKey : String,
               rankThreshold : Double,
               splitCriteria : String) = {
    val rec = new ModelRecord()

    SplitCriteria.get(splitCriteria) match {
      case Some(SplitCriteriaTypes.Classification) => {
        var numPos = 0.0
        var numNeg = 0.0
        for (example <- examples) {
          val label = if (example.get(rankKey).asScala.head._2 <= rankThreshold) false else true
          if (label) numPos += 1.0 else numNeg += 1.0
        }
        val sum = numPos + numNeg
        if (sum > 0.0) {
          // Convert from percentage positive to the -1 to 1 range
          val frac = numPos / sum
          rec.setFeatureWeight(2.0 * frac - 1.0)
        } else {
          rec.setFeatureWeight(0.0)
        }

      }
      case Some(SplitCriteriaTypes.Regression) => {
        var count: Double = 0.0
        var sum: Double = 0.0

        for (example <- examples) {
          val labelValue = example.get(rankKey).asScala.head._2

          count += 1.0
          sum += labelValue
        }

        // In regression case, leaf is the average of all the associated values
        rec.setFeatureWeight(sum / count)
      }
      case _ => {
        log.error("Unrecognized criteria type: %s".format(splitCriteria))
      }
    }

    rec
  }

  // Returns the best split if one exists.
  def getBestSplit(examples :  Array[util.Map[java.lang.String, util.Map[java.lang.String, java.lang.Double]]],
                   rankKey : String,
                   rankThreshold : Double,
                   numTries : Int,
                   minLeafCount : Int,
                   splitCriteria : String) : Option[ModelRecord] = {
    var bestRecord : Option[ModelRecord] = None
    var bestValue : Double = -1e10
    val rnd = new Random()
    for (i <- 0 until numTries) {
      // Pick an example index randomly
      val idx = rnd.nextInt(examples.size)
      val ex = examples(idx)
      val candidateOpt = getCandidateSplit(ex, rankKey, rnd)
      if (candidateOpt.isDefined) {
        val candidateValue = SplitCriteria.get(splitCriteria) match {
          case Some(SplitCriteriaTypes.Classification) => {
            evaluateClassificationSplit(
              examples, rankKey,
              rankThreshold,
              minLeafCount,
              splitCriteria, candidateOpt
            )
          }
          case Some(SplitCriteriaTypes.Regression) => {
            evaluateRegressionSplit(
              examples, rankKey,
              minLeafCount,
              splitCriteria, candidateOpt
            )
          }
          case _ =>
            log.error("Unrecognized split criteria: %s".format(splitCriteria))
            None
        }

        if (candidateValue.isDefined && candidateValue.get > bestValue) {
          bestValue = candidateValue.get
          bestRecord = candidateOpt
        }
      }
    }
    
    bestRecord
  }

  // Evaluate a classification-type split
  def evaluateClassificationSplit(
      examples : Array[util.Map[java.lang.String, util.Map[java.lang.String, java.lang.Double]]],
      rankKey : String,
      rankThreshold : Double,
      minLeafCount : Int,
      splitCriteria : String,
      candidateOpt : Option[ModelRecord]): Option[Double] = {
    var leftPos : Double = 0.0
    var rightPos : Double = 0.0
    var leftNeg : Double = 0.0
    var rightNeg : Double = 0.0

    for (example <- examples) {
      val response = BoostedStumpsModel.getStumpResponse(candidateOpt.get, example)
      val label = if (example.get(rankKey).asScala.head._2 <= rankThreshold) false else true
      if (response) {
        if (label) {
          rightPos += 1.0
        } else {
          rightNeg += 1.0
        }
      } else {
        if (label) {
          leftPos += 1.0
        } else {
          leftNeg += 1.0
        }
      }
    }
    val rightCount = rightPos + rightNeg
    val leftCount = leftPos + leftNeg

    if (rightCount >= minLeafCount && leftCount >= minLeafCount) {
      val p1 = rightPos / rightCount
      val n1 = rightNeg / rightCount
      val f1 = rightCount / (leftCount + rightCount)
      val p2 = leftPos / leftCount
      val n2 = leftNeg / leftCount
      val f2 = leftCount / (leftCount + rightCount)
      splitCriteria match {
        case "gini" => {
          // Using negative gini since we are maximizing.
          val gini = -(
              f1 * (p1 * (1.0 - p1) + n1 * (1.0 - n1)) +
                  f2 * (n2 * (1.0 - n2) + p2 * (1.0 - p2))
              )

          Some(gini)
        }
        case "information_gain" => {
          var ig = 0.0
          if (p1 > 0) {
            ig += f1 * p1 * scala.math.log(p1)
          }
          if (n1 > 0) {
            ig += f1 * n1 * scala.math.log(n1)
          }
          if (p2 > 0) {
            ig += f2 * p2 * scala.math.log(p2)
          }
          if (n2 > 0) {
            ig += f2 * n2 * scala.math.log(n2)
          }

          Some(ig)
        }
        case "hellinger" => {
          val scale = 1.0 / (leftCount * rightCount)
          // http://en.wikipedia.org/wiki/Bhattacharyya_distance
          val bhattacharyya = math.sqrt(leftPos * rightPos * scale) + math.sqrt(leftNeg * rightNeg * scale)
          // http://en.wikipedia.org/wiki/Hellinger_distance
          val hellinger = math.sqrt(1.0 - bhattacharyya)

          Some(hellinger)
        }
      }
    } else {
      None
    }
  }

  // Evaluate a regression-type split
  // See http://www.stat.cmu.edu/~cshalizi/350-2006/lecture-10.pdf for overview of algorithm used
  def evaluateRegressionSplit(
      examples : Array[util.Map[java.lang.String, util.Map[java.lang.String, java.lang.Double]]],
      rankKey : String,
      minLeafCount : Int,
      splitCriteria : String,
      candidateOpt : Option[ModelRecord]): Option[Double] = {
    var rightCount: Double = 0.0
    var rightMean: Double = 0.0
    var rightSumSq: Double = 0.0
    var leftCount: Double = 0.0
    var leftMean: Double = 0.0
    var leftSumSq: Double = 0.0

    for (example <- examples) {
      val response = BoostedStumpsModel.getStumpResponse(candidateOpt.get, example)
      val labelValue = example.get(rankKey).asScala.head._2

      // Using Welford's Method for computing mean and sum-squared errors in numerically stable way;
      // more details can be found in http://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance
      //
      // See unit test for verification that it is consistent with standard, two-pass approach
      if (response) {
        rightCount += 1
        val delta = labelValue - rightMean
        rightMean += delta / rightCount
        rightSumSq += delta * (labelValue - rightMean)
      } else {
        leftCount += 1
        val delta = labelValue - leftMean
        leftMean += delta / leftCount
        leftSumSq += delta * (labelValue - leftMean)
      }
    }

    if (rightCount >= minLeafCount && leftCount >= minLeafCount) {
      splitCriteria match {
        case "variance" =>
          Some(-(leftSumSq + rightSumSq))
      }
    } else {
      None
    }
  }

  // Returns a candidate split sampled from an example.
  def getCandidateSplit(ex : util.Map[java.lang.String, util.Map[java.lang.String, java.lang.Double]],
                        rankKey : String,
                        rnd : Random) : Option[ModelRecord] = {
    // Flatten the features and pick one randomly.
    val features = collection.mutable.ArrayBuffer[(String, String, Double)]()
    for (family <- ex) {
      if (!family._1.equals(rankKey)) {
        for (feature <- family._2) {
          features.append((family._1, feature._1, feature._2))
        }
      }
    }
    if (features.size == 0) {
      return None
    }
    val idx = rnd.nextInt(features.size)
    val rec = new ModelRecord()
    rec.setFeatureFamily(features(idx)._1)
    rec.setFeatureName(features(idx)._2)
    rec.setThreshold(features(idx)._3)

    Some(rec)
  }

  def trainAndSaveToFile(sc : SparkContext,
                         input : RDD[Example],
                         config : Config,
                         key : String) = {
    val model = train(sc, input, config, key)
    TrainingUtils.saveModel(model, config, key + ".model_output")
  }
}
