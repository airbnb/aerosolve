package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;

import java.util.HashSet;
import java.util.List;
import java.util.Map;

import com.typesafe.config.Config;

/**
 * "fields" specifies a list of float feature families to be deleted
 */
public class DeleteFloatFeatureFamilyTransform implements Transform {
  private List<String> fieldNames;

  @Override
  public void configure(Config config, String key) {
    fieldNames = config.getStringList(key + ".fields");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    if (floatFeatures == null) {
      return;
    }

    if (fieldNames == null) {
      return;
    }

    for (String fieldName: fieldNames) {
      Map<String, Double> feature = floatFeatures.get(fieldName);
      if (feature != null) {
        floatFeatures.remove(fieldName);
      }
    }
  }
}
