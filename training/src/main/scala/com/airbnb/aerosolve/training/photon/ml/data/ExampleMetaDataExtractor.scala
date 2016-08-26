package com.airbnb.aerosolve.training.photon.ml.data

import com.airbnb.aerosolve.core.Example

import scala.collection.immutable

/**
  * A trait defines a general pattern for extracting meta data from Example
  *
  */
trait ExampleMetaDataExtractor {

  /**
    * Build a meta data map given an example and optionally a context information map
    *
    * @param example An example
    * @param context The context info map
    */
  def buildMetaDataMap(example: Example,
                       context: immutable.Map[String, String]): java.util.Map[String, Long]
}
