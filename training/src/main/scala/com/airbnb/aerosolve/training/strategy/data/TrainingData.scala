package com.airbnb.aerosolve.training.strategy.data

import java.io.Serializable

import org.apache.spark.sql.Row


trait TrainingData [+T] extends Serializable {
  def parseSampleFromHiveRow(row: Row): T

  def selectData: String
}
