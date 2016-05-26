package com.airbnb.aerosolve.training.pipeline

import java.io.{BufferedReader, BufferedWriter, InputStreamReader, OutputStreamWriter}
import java.net.URI

import com.airbnb.aerosolve.core.Example
import com.airbnb.aerosolve.core.models.AbstractModel
import com.airbnb.aerosolve.core.transforms.Transformer
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.slf4j.{Logger, LoggerFactory}

/*
 * Miscellaneous utilities for pipeline file system access.
 */
object PipelineUtil {
  val log: Logger = LoggerFactory.getLogger("PipelineUtil")
  val hadoopConfiguration = new Configuration()

  def saveAndCommitAsTextFile[U](rdd: RDD[U], dest: String): Unit = {
    saveAndCommitAsTextFile(rdd, dest, 0)
  }

  def saveAndCommitAsTextFile[U](rdd: RDD[U], dest: String, overwrite: Boolean): Unit = {
    saveAndCommitAsTextFile(rdd, dest, 0, overwrite)
  }

  def hdfsFileExists(file: String): Boolean = {
    val hfs = FileSystem.get(
      new java.net.URI(file), new Configuration())

    val path = new Path(file)
    hfs.exists(path)
  }

  def saveAndCommitAsTextFile[U](
      rdd: RDD[U],
      dest: String,
      partition: Int,
      overwrite: Boolean = false): Unit = {
    log.info("Saving data to %s".format(dest))

    val hfs = FileSystem.get(
      new java.net.URI(dest), new Configuration())

    val tmpPath = new Path(dest + ".tmp")
    val destPath = new Path(dest)

    if (!hfs.exists(destPath) || overwrite) {
      try {
        if (hfs.exists(tmpPath)) {
          hfs.delete(tmpPath, true)
          log.info("deleted old tmp directory: " + tmpPath)
        }
        if (partition > 0) {
          // shuffle it to prevent imbalance output, which slows down the whole job.
          rdd.coalesce(partition, true)
            .saveAsTextFile(dest + ".tmp", classOf[GzipCodec])
        } else {
          rdd.saveAsTextFile(dest + ".tmp", classOf[GzipCodec])
        }
        if (hfs.exists(destPath)) {
          hfs.delete(destPath, true)
          log.info("deleted old directory: " + destPath)
        }
        log.info("committing " + dest)
        hfs.rename(tmpPath, destPath)
        log.info("committed " + dest)
      } catch {
        case e: Exception => {
          log.error("exception during save: ", e)
          log.info("deleting failed data for " + dest + ".tmp")
          hfs.delete(tmpPath, true)
          log.info("deleted " + dest + ".tmp")
          throw e
        }
      }
    } else {
      log.info("data already exists for " + dest)
    }
  }

  def writeStringToFile(str: String, output: String) = {
    val fs = FileSystem.get(new URI(output), hadoopConfiguration)
    val path = new Path(output)
    val stream = fs.create(path, true)
    val writer = new BufferedWriter(new OutputStreamWriter(stream))
    writer.write(str)
    writer.close()
  }

  def readStringFromFile(inputFile : String): String = {
    val fs = FileSystem.get(new URI(inputFile), hadoopConfiguration)
    val path = new Path(inputFile)
    val stream = fs.open(path)
    val reader = new BufferedReader(new InputStreamReader(stream))
    val str = Stream.continually(reader.readLine()).takeWhile(_ != null).mkString("\n")
    str
  }

  def copyFiles(srcPath: String, destPath: String, deleteSource: Boolean = false) = {
    val src = new Path(srcPath)
    val dest = new Path(destPath)
    val fsConfig = new Configuration()
    val fs = FileSystem.get(new java.net.URI(destPath), fsConfig)
    log.info("Copying successful from " + src + " to " + dest)
    try {
      FileUtil.copy(fs, src, fs, dest, deleteSource, fsConfig)
    } catch {
      case e: Exception => {
        log.info("Copy failed " + dest)
        System.exit(-1)
      }
    }
    log.info("Copy done.")
  }

  def scoreExamples(sc: SparkContext,
                    transformer: Transformer,
                    modelOpt: AbstractModel,
                    examples: RDD[Example],
                    isTraining: Example => Boolean,
                    labelKey: String): RDD[(Float, String)] = {
    val modelBC = sc.broadcast(modelOpt)
    val transformerBC = sc.broadcast(transformer)
    val scoreAndLabel = examples
      .map(example => {
        transformerBC.value.combineContextAndItems(example)
        val score = modelBC.value.scoreItem(example.example.get(0))
        val rank = example.example.get(0).floatFeatures.get(labelKey).get("")
        val label = (if (isTraining(example)) "TRAIN_" else "HOLD_") + (if (rank > 0) "P" else "N")
        (score, label)
      })
    scoreAndLabel
  }
}
