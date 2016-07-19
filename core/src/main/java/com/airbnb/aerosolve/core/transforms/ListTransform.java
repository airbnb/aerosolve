package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;

import com.typesafe.config.Config;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Created by hector_yee on 8/25/14.
 * A transform that accepts a list of other transforms and applies them as a group
 * in the order specified by the list.
 */
public class ListTransform implements Transform {
  private List<Transform> transforms;

  @Override
  public void configure(Config config, String key) {
    transforms = new ArrayList<>();
    List<String> transformKeys = config.getStringList(key + ".transforms");
    for (String transformKey : transformKeys) {
      Transform tmpTransform = TransformFactory.createTransform(config, transformKey);
      if (tmpTransform != null) {
        transforms.add(tmpTransform);
      }
    }
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    for (Transform transform : transforms) {
      transform.doTransform(featureVector);
    }
  }

  @Override
  public void doTransform(Stream<FeatureVector> featureVectors) {
    // collect stream because they can only be operated once
    List<FeatureVector> vectors = featureVectors.collect(Collectors.toList());
    for (Transform transform : transforms) {
      transform.doTransform(vectors.stream());
    }
  }
}
