package com.airbnb.aerosolve.training.photon.ml.data

import com.airbnb.aerosolve.core.Example

import scala.collection.JavaConverters._
import scala.collection.immutable


/**
  * An implementation of [[ExampleMetaDataExtractor]] that looks into one single float feature
  * family of [[Example]] and cast all values into non-negative longs.
  *
  */
class SingleFloatFamilyMetaExtractor
  extends ExampleMetaDataExtractor {

  override def buildMetaDataMap(example: Example,
                                context: immutable.Map[String, String]): java.util.Map[String, Long] = {
    val meta = new java.util.HashMap[String, Long]()
    val metaFamily = context.getOrElse("metaFamily", "meta")

    val f = example.getExample().get(0).floatFeatures

    if (!f.containsKey(metaFamily)) {
      throw new IllegalArgumentException(
        s"meta data feature field: [${metaFamily}] not presented in " +
          s"the example's float features: ${example.toString()}")
    }

    f.get(metaFamily).asScala.foreach { case (name, value) =>
      meta.put(name, Math.abs(value.toLong))
    }

    meta
  }
}
