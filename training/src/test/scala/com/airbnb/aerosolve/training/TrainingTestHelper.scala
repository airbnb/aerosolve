package com.airbnb.aerosolve.training

import java.util

import com.airbnb.aerosolve.core.{Example, FeatureVector, FunctionForm}
import com.airbnb.aerosolve.core.models.{SplineModel, AdditiveModel}
import com.airbnb.aerosolve.core.util.{Spline, Linear}
import org.slf4j.LoggerFactory
import scala.collection.mutable.ArrayBuffer
import scala.collection.JavaConverters._

object TrainingTestHelper {
  val log = LoggerFactory.getLogger("TrainingTestHelper")

  def makeExample(x : Double,
                  y : Double,
                  target : Double) : Example = {
    val example = new Example
    val item: FeatureVector = new FeatureVector
    item.setFloatFeatures(new java.util.HashMap)
    item.setStringFeatures(new java.util.HashMap)
    val floatFeatures = item.getFloatFeatures
    val stringFeatures = item.getStringFeatures
    // A string feature that is always on.
    stringFeatures.put("BIAS", new java.util.HashSet)
    stringFeatures.get("BIAS").add("B")
    // A string feature that is sometimes on
    if (x + y < 0) {
      stringFeatures.put("NEG", new java.util.HashSet)
      stringFeatures.get("NEG").add("T")
    }
    floatFeatures.put("$rank", new java.util.HashMap)
    floatFeatures.get("$rank").put("", target)
    floatFeatures.put("loc", new java.util.HashMap)
    val loc = floatFeatures.get("loc")
    loc.put("x", x)
    loc.put("y", y)
    example.addToExample(item)
    example
  }

  def makeSimpleClassificationExamples = {
    val examples = ArrayBuffer[Example]()
    val label = ArrayBuffer[Double]()
    val rnd = new java.util.Random(1234)
    var numPos: Int = 0
    for (i <- 0 until 200) {
      val x = 2.0 * rnd.nextDouble() - 1.0
      val y = 10.0 * (2.0 * rnd.nextDouble() - 1.0)
      val poly = x + y
      val rank = if (poly < 1.0) {
        1.0
      } else {
        -1.0
      }
      if (rank > 0) numPos = numPos + 1
      label += rank
      examples += makeExample(x, y, rank)
    }
    (examples, label, numPos)
  }

  def makeMulticlassExample(x : Double,
                            y : Double,
                            z : Double,
                            label : String) : Example = {
    val example = new Example
    val item: FeatureVector = new FeatureVector
    item.setFloatFeatures(new java.util.HashMap)
    item.setStringFeatures(new java.util.HashMap)
    val floatFeatures = item.getFloatFeatures
    val stringFeatures = item.getStringFeatures
    // A string feature that is always on.
    stringFeatures.put("BIAS", new java.util.HashSet)
    stringFeatures.get("BIAS").add("B")
    floatFeatures.put("$rank", new java.util.HashMap)
    floatFeatures.get("$rank").put(label, 1.0)
    floatFeatures.put("loc", new java.util.HashMap)
    val loc = floatFeatures.get("loc")
    loc.put("x", x)
    loc.put("y", y)
    loc.put("z", z)
    example.addToExample(item)
    example
  }

  def makeSimpleMulticlassClassificationExamples = {
    val examples = ArrayBuffer[Example]()
    val labels = ArrayBuffer[String]()
    val rnd = new java.util.Random(1234)
    for (i <- 0 until 200) {
      var x = 2.0 * rnd.nextDouble() - 1.0
      var y = 2.0 * rnd.nextDouble() - 1.0
      val z = 2.0 * rnd.nextDouble() - 1.0
      var label : String = ""
      rnd.nextInt(4) match {
        case 0 => {
          label = "top_left"
          x = x - 10.0
          y = y + 10.0
        }
        case 1 => {
          label = "top_right"
          x = x + 10.0
          y = y + 10.0
        }
        case 2 => {
          label = "bot_left"
          x = x - 10.0
          y = y - 10.0
        }
        case 3 => {
          label = "bot_right"
          x = x + 10.0
          y = y - 10.0
        }
      }
      labels += label
      examples += makeMulticlassExample(x, y, z, label)
    }
    (examples, labels)
  }

  def makeHybridExample(x : Double,
                        y : Double,
                        target : Double) : Example = {
    val example = new Example
    val item: FeatureVector = new FeatureVector
    item.setFloatFeatures(new java.util.HashMap)
    val floatFeatures = item.getFloatFeatures
    floatFeatures.put("$rank", new java.util.HashMap)
    floatFeatures.get("$rank").put("", target)
    floatFeatures.put("loc", new java.util.HashMap)
    floatFeatures.put("xy", new util.HashMap)
    val loc = floatFeatures.get("loc")
    loc.put("x", x)
    loc.put("y", y)
    val xy = floatFeatures.get("xy")
    xy.put("xy", x * y)
    example.addToExample(item)
    example
  }

  def makeClassificationExamples = {
    val examples = ArrayBuffer[Example]()
    val label = ArrayBuffer[Double]()
    val rnd = new java.util.Random(1234)
    var numPos : Int = 0
    for (i <- 0 until 200) {
      val x = 2.0 * rnd.nextDouble() - 1.0
      val y = 10.0 * (2.0 * rnd.nextDouble() - 1.0)
      val poly = x * x + 0.1 * y * y + 0.1 * x + 0.2 * y - 0.1 + Math.sin(x)
      val rank = if (poly < 1.0) {
        1.0
      } else {
        -1.0
      }
      if (rank > 0) numPos = numPos + 1
      label += rank
      examples += makeExample(x, y, rank)
    }
    (examples, label, numPos)
  }

  def makeLinearClassificationExamples = {
    val examples = ArrayBuffer[Example]()
    val label = ArrayBuffer[Double]()
    val rnd = new java.util.Random(1234)
    var numPos : Int = 0
    for (i <- 0 until 200) {
      val x = 2.0 * rnd.nextDouble() - 1.0
      val y = 10.0 * (2.0 * rnd.nextDouble() - 1.0)
      val linear = -6.0 * x + y + 3.0 + 2 * x * y
      val rank = if (linear < 1.0) {
        1.0
      } else {
        -1.0
      }
      if (rank > 0) numPos = numPos + 1
      label += rank
      examples += makeHybridExample(x, y, rank)
    }
    (examples, label, numPos)
  }

  def makeRegressionExamples(randomSeed: Int = 1234) = {
    val examples = ArrayBuffer[Example]()
    val label = ArrayBuffer[Double]()
    val rnd = new java.util.Random(randomSeed)

    for (i <- 0 until 200) {
      val x = 4.0 * (2.0 * rnd.nextDouble() - 1.0)
      val y = 4.0 * (2.0 * rnd.nextDouble() - 1.0)

      // Curve will be a "saddle" with flat regions where, for instance, x = 0 and y > 2.06 or y < -1.96
      val flattenedQuadratic = math.max(x * x - 2 * y * y - 0.5 * x + 0.2 * y, -8.0)

      examples += makeExample(x, y, flattenedQuadratic)
      label += flattenedQuadratic
    }

    (examples, label)
  }

  def makeLinearRegressionExamples(randomSeed: Int = 1234) = {
    val examples = ArrayBuffer[Example]()
    val label = ArrayBuffer[Double]()
    val rnd = new java.util.Random(randomSeed)

    for (i <- 0 until 200) {
      val x = 2.0 * (rnd.nextDouble() - 0.5)
      val y = 2.0 * (rnd.nextDouble() - 0.5)
      val z = 0.1 * x * y - 0.5 * x + 0.2 * y + 1.0
      examples += makeHybridExample(x, y, z)
      label += z
    }

    (examples, label)
  }

  def printSpline(model: SplineModel) = {
    val weights = model.getWeightSpline.asScala
    for (familyMap <- weights) {
      for (featureMap <- familyMap._2.asScala) {
        log.info(("family=%s,feature=%s,"
                  + "minVal=%f, maxVal=%f, weights=%s")
                   .format(familyMap._1,
                           featureMap._1,
                           featureMap._2.spline.getMinVal,
                           featureMap._2.spline.getMaxVal,
                           featureMap._2.spline.getWeights.mkString(",")
          )
        )
      }
    }
  }

  def printAdditiveModel(model: AdditiveModel) = {
    val weights = model.getWeights.asScala
    for (familyMap <- weights) {
      for (featureMap <- familyMap._2.asScala) {
        log.info("family=%s,feature=%s".format(familyMap._1, featureMap._1))
        val func = featureMap._2
        val funcForm = func.getFunctionForm
        log.info("functionForm=%s".format(funcForm))
        log.info("minVal=%f, maxVal=%f, weights=%s"
                   .format(func.getMinVal,
                           func.getMaxVal,
                           func.getWeights.mkString(","))
        )
      }
    }
  }
}