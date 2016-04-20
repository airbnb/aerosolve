package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;

import java.util.Map;
import java.util.Set;

import com.typesafe.config.Config;

/**
 * "field1" specifies the string column to be deleted
 */
public class DeleteStringFeatureFamilyTransform implements Transform {
  private String fieldName1;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    if (stringFeatures == null) {
      return ;
    }

    Set<String> feature1 = stringFeatures.get(fieldName1);
    if (feature1 == null) {
      return;
    }

    stringFeatures.remove(fieldName1);
  }
}
