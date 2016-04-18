package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;

import java.io.Serializable;

/**
 * Created by hector_yee on 8/25/14.
 * Base class for feature transforms.
 */
public interface Transform extends Serializable {
  // Configure the transform from the supplied config and key.
  void configure(Config config, String key);

  // Applies a transform to the featureVector.
  void doTransform(FeatureVector featureVector);
}
