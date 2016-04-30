package com.airbnb.aerosolve.demo.ImageImpressionism

import java.awt.image.BufferedImage
import java.util
import javax.imageio.ImageIO

import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.core.ModelRecord
import com.airbnb.aerosolve.core.FeatureVector
import com.airbnb.aerosolve.core.models.AbstractModel
import com.airbnb.aerosolve.core.models.LinearModel
import com.airbnb.aerosolve.core.util.Util
import com.airbnb.aerosolve.core.transforms.Transformer
import com.airbnb.aerosolve.training.TrainingUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.spark.{SparkContext, SparkConf}
import org.slf4j.{LoggerFactory, Logger}
import com.typesafe.config.Config
import com.typesafe.config.ConfigFactory
import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer
import org.apache.hadoop.io.compress.GzipCodec

object ImageImpressionismPipeline {
  val log: Logger = LoggerFactory.getLogger("ImageImpressionismPipeline")

  case class Pixel(x : Int, y : Int, r : Double, g : Double, b : Double)

  def makeTrainingRun(sc : SparkContext, config : Config) = {
    val imageName : String = config.getString("input")
    val output : String = config.getString("output")
    log.info("Reading image %s".format(imageName))
    log.info("Writing training data to %s".format(output))

    val image = readImage(imageName)
    val pixels = ArrayBuffer[Pixel]()
    for (x <- 0 until image.getWidth) {
      for (y <- 0 until image.getHeight) {
        val rgb = image.getRGB(x, y)
        val r = ((rgb >> 16) & 0xff) / 255.0
        val g = ((rgb >> 8) & 0xff) / 255.0
        val b = ((rgb >> 0) & 0xff) / 255.0
        val pixel = Pixel(x, y, r, g, b)
        pixels.add(pixel)
      }
    }
    sc.parallelize(pixels)
      .map(x => pixelToExample(x, true))
      .map(Util.encode)
      .saveAsTextFile(output, classOf[GzipCodec])
  }

  def trainModel(sc : SparkContext, config : Config) = {
    val trainConfig = config.getConfig("train_model")
    val trainingDataName = trainConfig.getString("input")
    val modelKey = trainConfig.getString("modelKey")
    log.info("Training on %s".format(trainingDataName))

    val input = sc.textFile(trainingDataName).map(Util.decodeExample).filter(isTraining)
    TrainingUtils.trainAndSaveToFile(sc, input, config, modelKey)
  }

  def makeImpression(sc : SparkContext, config : Config) = {
    val impConfig = config.getConfig("make_impression")
    val modelKey = impConfig.getString("modelKey")
    val output = impConfig.getString("output")

    val width : Int = impConfig.getInt("width")
    val height : Int = impConfig.getInt("height")

    val modelName = config.getConfig(modelKey).getString("model_output")
    // Load the model
    val modelOpt = TrainingUtils.loadScoreModel(modelName)
    if (modelOpt == None) {
      log.error("Could not open %s".format(modelName))
      System.exit(-1)
    }

    val model = modelOpt.get

    // Make the transformer
    val transformer = new Transformer(config, modelKey)

    val outImage = scoreImage(width, height, model, transformer)
    log.info("Saving to %s".format(output))
    ImageIO.write(outImage, "jpg", new java.io.File(output))
  }

  def makeMovie(sc : SparkContext, config : Config) = {
    val impConfig = config.getConfig("make_movie")
    val modelKey = impConfig.getString("modelKey")
    val output = impConfig.getString("output")

    val width : Int = impConfig.getInt("width")
    val height : Int = impConfig.getInt("height")
    val numFrames : Int = impConfig.getInt("numFrames")

    val modelName = config.getConfig(modelKey).getString("model_output")
    // Load the model, this time using low level access in order to order by stumps.
    val weights = sc
      .textFile(modelName)
      .map(Util.decodeModel)
      .filter(x => x.featureName != null)
      .collect
      .toArray

    log.info("Total weights = %d".format(weights.length))

    // Make the transformer
    val transformer = new Transformer(config, modelKey)

    for (i <- 0 until numFrames) {
      val model = new LinearModel()
      val frac = i.toDouble / (numFrames.toDouble - 1.0)
      addRecords(weights, model, frac)
      val name = output.format(i)
      log.info("Saving to %s".format(name))
      val outImage = scoreImage(width, height, model, transformer)
      ImageIO.write(outImage, "jpg", new java.io.File(name))
    }
  }

  def addRecords(weights : Array[ModelRecord],
                 linear : LinearModel,
                 frac : Double): Unit = {
    val count = math.max(1, (frac * weights.length).toInt)
    val candidates = weights.take(count)
    val wt : util.Map[java.lang.String, util.Map[java.lang.String, java.lang.Float]] =
      new util.HashMap()
    linear.setWeights(wt)
    for (rec <- candidates) {
      var family : util.Map[java.lang.String, java.lang.Float] = wt.get(rec.featureFamily)
      if (family == null) {
        family = new util.HashMap()
        wt.put(rec.featureFamily, family)
      }
      family.put(rec.featureName, rec.featureWeight.toFloat)
    }
  }


  def scoreImage(width : Int, height : Int, model : AbstractModel, transformer : Transformer) = {
    val outImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    log.info("Scoring model")
    for (x <- 0 until width) {
      for (y <- 0 until height) {
        val pixel = Pixel(x, y, 0, 0, 0)
        val examples = pixelToExample(pixel, false)
        transformer.combineContextAndItems(examples)
        val redScore = math.max(0.0, math.min(1.0, model.scoreItem(examples.example(0))))
        val greenScore = math.max(0.0, math.min(1.0, model.scoreItem(examples.example(1))))
        val blueScore = math.max(0.0, math.min(1.0, model.scoreItem(examples.example(2))))
        val r : Int = (redScore * 255.0).toInt
        val g : Int = (greenScore * 255.0).toInt
        val b : Int = (blueScore * 255.0).toInt
        outImage.setRGB(x, y, (r << 16) | (g << 8) | b)
      }
    }
    outImage
  }

  // Emits three examples, one for each color channel.
  def pixelToExample(pix : Pixel, withLabel : Boolean) : Example = {
    val result = new Example()

    // This is shared data for all three channels so we put it in "context"
    val context = Util.createNewFeatureVector()
    val loc = Util.getOrCreateFloatFeature("LOC", context.floatFeatures)
    loc.put("X", pix.x)
    loc.put("Y", pix.y)

    // Create examples for each channel
    val red = Util.createNewFeatureVector()
    // What channel is this example for
    val redChannel = Util.getOrCreateStringFeature("C", red.stringFeatures)
    redChannel.add("Red")

    val green = Util.createNewFeatureVector()
    // What channel is this example for
    val greenChannel = Util.getOrCreateStringFeature("C", green.stringFeatures)
    greenChannel.add("Green")

    val blue = Util.createNewFeatureVector()
    // What channel is this example for
    val blueChannel = Util.getOrCreateStringFeature("C", blue.stringFeatures)
    blueChannel.add("Blue")

    if (withLabel) {
      val redIntensity = Util.getOrCreateFloatFeature("$target", red.floatFeatures)
      redIntensity.put("", pix.r)
      val greenIntensity = Util.getOrCreateFloatFeature("$target", green.floatFeatures)
      greenIntensity.put("", pix.g)
      val blueIntensity = Util.getOrCreateFloatFeature("$target", blue.floatFeatures)
      blueIntensity.put("", pix.b)
    }

    result.example = new util.ArrayList[FeatureVector]()
    result.context = context
    result.example.add(red)
    result.example.add(green)
    result.example.add(blue)

    return result
  }

  def readImage(imageName : String) : BufferedImage = {
    val path = new Path(imageName)
    val fileSystem = FileSystem.get(new java.net.URI(imageName), new Configuration())
    val instream = fileSystem.open(path)
    ImageIO.read(instream)
  }

  def isTraining(examples : Example) : Boolean = {
    // Take the hash code mod 255 and keep the first 16 as holdout.
    (examples.toString.hashCode & 0xFF) > 16
  }
}