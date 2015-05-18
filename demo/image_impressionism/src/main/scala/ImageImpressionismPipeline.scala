package com.airbnb.aerosolve.demo.ImageImpressionism;

import java.awt.image.BufferedImage
import javax.imageio.ImageIO

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.spark.{SparkContext, SparkConf}
import org.slf4j.{LoggerFactory, Logger}
import com.typesafe.config.Config
import com.typesafe.config.ConfigFactory
import scala.collection.JavaConversions._

object ImageImpressionismPipeline {
  val log: Logger = LoggerFactory.getLogger("ImageImpressionismPipeline")

  def makeTrainingRun(sc : SparkContext, config : Config) = {
    val imageName : String = config.getString("input")
    val output : String = config.getString("output")
    log.info("Reading image %s".format(imageName))
    log.info("Writing training data to %s".format(output))

    val path = new Path(imageName)
    val fileSystem = FileSystem.get(new java.net.URI(imageName), new Configuration())
    val instream = fileSystem.open(path)
    val image = ImageIO.read(instream)
  }
}