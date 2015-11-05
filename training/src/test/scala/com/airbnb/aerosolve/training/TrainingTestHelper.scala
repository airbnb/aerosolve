package com.airbnb.aerosolve.training

import com.airbnb.aerosolve.core.{Example, FeatureVector}
import com.airbnb.aerosolve.core.models.SplineModel
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
    val floatFeatures = item.getFloatFeatures
    floatFeatures.put("$rank", new java.util.HashMap)
    floatFeatures.get("$rank").put("", target)
    floatFeatures.put("loc", new java.util.HashMap)
    val loc = floatFeatures.get("loc")
    loc.put("x", x)
    loc.put("y", y)
    example.addToExample(item)
    example
  }

  private def makeExamples(number : Int, makeLabel : (Double) => Double) = {
    val examples = ArrayBuffer[Example]()
    val labels = ArrayBuffer[Double]()
    val rnd = new java.util.Random(1234)
    var numPos : Int = 0
    for (i <- 0 until number) {
      val x = 2.0 * rnd.nextDouble() - 1.0
      val y = 10.0 * (2.0 * rnd.nextDouble() - 1.0)
      val poly = x * x + 0.1 * y * y + 0.1 * x + 0.2 * y - 0.1 + Math.sin(x)
      val label = makeLabel(poly)
      if (label > 0) numPos = numPos + 1
      labels += label
      examples += makeExample(x, y, label)
    }
    (examples, labels, numPos)
  }

  private def reverseBinLabel(label : Double) : Double = {
    if (label < 1.0) {
      1.0
    } else {
      -1.0
    }
  }

  private def binLabel(label : Double) : Double = {
    if (label >= 0) {
      1.0
    } else {
      -1.0
    }
  }

  def makeClassificationExamples = {
    makeExamples(200, reverseBinLabel)
  }

  def makeRegressionAndClassificationExamples(number : Int, randomSeed: Int = 1234) = {
    val cExamples = ArrayBuffer[Example]()
    val clabels = ArrayBuffer[Double]()
    val rExamples = ArrayBuffer[Example]()
    val rlabels = ArrayBuffer[Double]()
    val rnd = new java.util.Random(1234)
    var numPos : Int = 0
    for (i <- 0 until number) {
      val x = 2.0 * rnd.nextDouble() - 1.0
      val y = 10.0 * (2.0 * rnd.nextDouble() - 1.0)
      val poly = x * x + 0.1 * y * y + 0.1 * x + 0.2 * y - 0.1 + Math.sin(x)
      val label = binLabel(poly)
      if (label > 0) numPos = numPos + 1
      clabels += label
      rlabels += poly
      cExamples += makeExample(x, y, label)
      rExamples += makeExample(x, y, poly)
    }
    (rExamples, rlabels, cExamples, clabels, numPos)
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
}