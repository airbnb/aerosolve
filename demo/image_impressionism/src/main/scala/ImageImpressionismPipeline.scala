package com.airbnb.aerosolve.demo.ImageImpressionism;

import java.awt.image.BufferedImage
import java.util
import javax.imageio.ImageIO

import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.core.FeatureVector
import com.airbnb.aerosolve.core.util.Util
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.spark.{SparkContext, SparkConf}
import org.slf4j.{LoggerFactory, Logger}
import com.typesafe.config.Config
import com.typesafe.config.ConfigFactory
import scala.collection.JavaConversions._
import scala.collection.mutable.ArrayBuffer

object ImageImpressionismPipeline {
  val log: Logger = LoggerFactory.getLogger("ImageImpressionismPipeline")

  case class Pixel(x : Int, y : Int, r : Double, g : Double, b : Double)

  def makeTrainingRun(sc : SparkContext, config : Config) = {
    val imageName : String = config.getString("input")
    val output : String = config.getString("output")
    log.info("Reading image %s".format(imageName))
    log.info("Writing training data to %s".format(output))

    val path = new Path(imageName)
    val fileSystem = FileSystem.get(new java.net.URI(imageName), new Configuration())
    val instream = fileSystem.open(path)
    val image : BufferedImage = ImageIO.read(instream)
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
      .map(pixelToExample)
      .map(Util.encode)
      .saveAsTextFile(output)
  }

  // Emits three examples, one for each color channel.
  def pixelToExample(pix : Pixel) : Example = {
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
    // Regression target for this channel
    val redIntensity = Util.getOrCreateFloatFeature("$target", red.floatFeatures)
    redIntensity.put("", pix.r)

    val green = Util.createNewFeatureVector()
    // What channel is this example for
    val greenChannel = Util.getOrCreateStringFeature("C", green.stringFeatures)
    greenChannel.add("Red")
    // Regression target for this channel
    val greenIntensity = Util.getOrCreateFloatFeature("$target", green.floatFeatures)
    greenIntensity.put("", pix.g)

    val blue = Util.createNewFeatureVector()
    // What channel is this example for
    val blueChannel = Util.getOrCreateStringFeature("C", blue.stringFeatures)
    blueChannel.add("Red")
    // Regression target for this channel
    val blueIntensity = Util.getOrCreateFloatFeature("$target", blue.floatFeatures)
    blueIntensity.put("", pix.b)

    result.example.add(red)
    result.example.add(green)
    result.example.add(blue)

    return result
  }
}