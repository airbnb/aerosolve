package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.typesafe.config.Config;

/**
 * "fields" specifies a list of string feature families to be deleted
 */
public class DeleteStringFeatureFamilyTransform implements Transform {
  private List<String> fieldNames;

  @Override
  public void configure(Config config, String key) {
    fieldNames = config.getStringList(key + ".fields");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    if (stringFeatures == null) {
      return;
    }

    if (fieldNames == null) {
      return;
    }

    for (String fieldName: fieldNames) {
      Set<String> feature = stringFeatures.get(fieldName);
      if (feature != null) {
        stringFeatures.remove(fieldName);
      }
    }
  }
}
