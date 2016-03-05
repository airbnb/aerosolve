package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import java.util.Map;
import com.typesafe.config.Config;

/**
 * "field1" specifies the float feature family to be deleted
 */

public class DeleteFloatFeatureFamilyTransform extends Transform {
  private String fieldName1;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    if (floatFeatures == null) {
      return ;
    }

    Map<String, Double> feature1 = floatFeatures.get(fieldName1);
    if (feature1 == null) {
      return;
    }

    floatFeatures.remove(fieldName1);
  }
}