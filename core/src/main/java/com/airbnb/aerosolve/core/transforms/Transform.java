package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;

import com.typesafe.config.Config;

import java.io.Serializable;
import java.util.stream.Stream;

/**
 * Created by hector_yee on 8/25/14.
 * Base class for feature transforms.
 */
public interface Transform extends Serializable {
  /**
   * Configure the transform from the supplied config and key. <p> This is where initialization
   * should take place. Ideally we want this to be a constructor instead or use a builder pattern.
   */
  void configure(Config config, String key);

  /**
   * Apply this transform to a single feature vector.
   */
  void doTransform(FeatureVector featureVector);

  /**
   * Applies this transform to a series of featureVector.
   *
   * @implNote this function can be overridden if the transform can be applied much more efficiency
   * in (small) batches If such implementation exists, one would typically override the single
   * feature vector implementation with the following instead:
   * <pre> <code>
   *  @Override
   *  public void doTransform(FeatureVector featureVector) {
   *    doTransform(Stream.of(featureVector));
   *  }
   * </code> </pre>
   */
  default void doTransform(Iterable<FeatureVector> featureVectors) {
    featureVectors.forEach(this::doTransform);
  }
}
