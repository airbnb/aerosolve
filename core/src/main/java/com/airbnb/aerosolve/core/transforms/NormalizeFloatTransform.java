package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;

import java.util.Map;

// L2 normalizes a float feature
public class NormalizeFloatTransform implements Transform {
  private String fieldName1;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
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

    double norm = 0.0;
    for (Map.Entry<String, Double> feat : feature1.entrySet()) {
      norm += feat.getValue() * feat.getValue();
    }
    if (norm > 0.0) {
      double scale = 1.0 / Math.sqrt(norm);
      for (Map.Entry<String, Double> feat : feature1.entrySet()) {
        feat.setValue(feat.getValue() * scale);
      }
    }
  }
}
