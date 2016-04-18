package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;
import java.util.List;
import java.util.Map;

public class DeleteFloatFeatureTransform implements Transform {
  private String fieldName1;
  private List<String> keys;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    keys = config.getStringList(key + ".keys");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();

    if (floatFeatures == null) {
      return;
    }
    Map<String, Double> feature1 = floatFeatures.get(fieldName1);
    if (feature1 == null) {
      return;
    }

    for (String key : keys) {
      feature1.remove(key);
    }
  }
}
