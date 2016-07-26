package com.airbnb.aerosolve.training.pipeline

import com.airbnb.aerosolve.training._

/*
 * Writes to result file
 */
object ResultUtil {
  def writeResults(metrics: Array[(String, Double)], resultOutputPath: String) = {
    // Put into a one line json format
    val elements = metrics.map{ x =>
      val metricStr = if (x._1 contains "THRESHOLD") x._1 else x.toString
      val line = "\'" + metricStr + "\'"
      line
    }
    val json = elements.mkString("[", ",", "]")
    PipelineUtil.writeStringToFile(json, resultOutputPath)
  }
}
