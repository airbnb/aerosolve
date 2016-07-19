package com.airbnb.aerosolve.training

import java.util

import com.airbnb.aerosolve.core.models.{AdditiveModel, SplineModel}
import com.airbnb.aerosolve.core.{Example, FeatureVector}
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

object TrainingTestHelper {
  val log = LoggerFactory.getLogger("TrainingTestHelper")

  def makeExample(x: Double,
                  y: Double,
                  target: Double): Example = {
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

  def makeDenseExample(x: Double,
                       y: Double,
                       target: Double): Example = {
    val example = new Example
    val item: FeatureVector = new FeatureVector
    item.setDenseFeatures(new java.util.HashMap)
    val denseFeatures = item.getDenseFeatures
    denseFeatures.put("d", java.util.Arrays.asList(x, y))
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

  def makeMulticlassExample(x: Double,
                            y: Double,
                            z: Double,
                            label: (String, Double),
                            label2: Option[(String, Double)]): Example = {
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
    floatFeatures.get("$rank").put(label._1, label._2)
    if (label2.isDefined) {
      floatFeatures.get("$rank").put(label2.get._1, label2.get._2)
    }
    floatFeatures.put("loc", new java.util.HashMap)
    val loc = floatFeatures.get("loc")
    loc.put("x", x)
    loc.put("y", y)
    loc.put("z", z)
    example.addToExample(item)
    example
  }

  def makeSimpleMulticlassClassificationExamples(multiLabel: Boolean) = {
    val examples = ArrayBuffer[Example]()
    val labels = ArrayBuffer[String]()
    val rnd = new java.util.Random(1234)
    for (i <- 0 until 1000) {
      var x = 2.0 * rnd.nextDouble() - 1.0
      var y = 2.0 * rnd.nextDouble() - 1.0
      val z = 2.0 * rnd.nextDouble() - 1.0
      var label: String = ""
      rnd.nextInt(4) match {
        case 0 =>
          label = "top_left"
          x = x - 10.0
          y = y + 10.0
        case 1 =>
          label = "top_right"
          x = x + 10.0
          y = y + 10.0
        case 2 =>
          label = "bot_left"
          x = x - 10.0
          y = y - 10.0
        case 3 =>
          label = "bot_right"
          x = x + 10.0
          y = y - 10.0
      }
      labels += label
      if (multiLabel) {
        val label2 = if (x > 0) "right" else "left"
        examples += makeMulticlassExample(x, y, z, (label, 1.0), Some((label2, 0.1)))
      } else {
        examples += makeMulticlassExample(x, y, z, (label, 1.0), None)
      }
    }
    (examples, labels)
  }

  def makeNonlinearMulticlassClassificationExamples() = {
    val examples = ArrayBuffer[Example]()
    val labels = ArrayBuffer[String]()
    val rnd = new java.util.Random(1234)
    for (i <- 0 until 1000) {
      val x = 20.0 * rnd.nextDouble() - 10.0
      val y = 20.0 * rnd.nextDouble() - 10.0
      val z = 20.0 * rnd.nextDouble() - 10.0
      val d = math.sqrt(x * x + y * y + z * z)
      // Three nested layers inner, middle and outer
      val label: String = if (d < 5) "inner"
      else if (d < 10) "middle" else "outer"
      labels += label
      examples += makeMulticlassExample(x, y, z, (label, 1.0), None)
    }
    (examples, labels)
  }

  def makeHybridExample(x: Double,
                        y: Double,
                        target: Double): Example = {
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

  lazy val makeClassificationExamples = {
    val examples = ArrayBuffer[Example]()
    val label = ArrayBuffer[Double]()
    val rnd = new java.util.Random(1234)
    var numPos: Int = 0
    for (i <- 0 until 500) {
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

  lazy val makeLinearClassificationExamples = {
    val examples = ArrayBuffer[Example]()
    val label = ArrayBuffer[Double]()
    val rnd = new java.util.Random(1234)
    var numPos: Int = 0
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

    for (i <- 0 until 400) {
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
        log.info(func.toString)
      }
    }
  }
}
