package com.airbnb.aerosolve.training.pipeline

import java.io.{BufferedWriter, OutputStreamWriter}
import java.net.URI

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem

/*
 * Writes to result file
 */
object ResultUtil {
  def writeResults(metrics: Array[(String, Double)], resultOutputPath: String) = {
    val fileSystem = FileSystem.get(new java.net.URI(result), new Configuration())
    val file = fileSystem.create(new Path(result), true)
    val writer = new BufferedWriter(new OutputStreamWriter(file))
    metrics.foreach{ x => 
      writer.write(if (x._1 contains "THRESHOLD") x._1 else x.toString)
      writer.newLine()
    }
    writer.flush()
    writer.close()
    file.close()
  }
}
