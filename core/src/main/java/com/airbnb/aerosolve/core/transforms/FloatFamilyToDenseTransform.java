package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.util.Arrays;
import java.util.Map;

/**
 * cross one float family with the other and out a 2D dense feature.
 */
public class FloatFamilyToDenseTransform implements Transform {
  private String fieldName1;
  private String fieldName2;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    fieldName2 = config.getString(key + ".field2");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();

    if (floatFeatures == null) {
      return;
    }

    Map<String, Double> map1 = floatFeatures.get(fieldName1);
    if (map1 == null || map1.isEmpty()) return;

    Map<String, Double> map2 = floatFeatures.get(fieldName2);
    if (map2 == null || map2.isEmpty()) return;

    cross(map1, map2, featureVector);
  }

  static void cross(
      Map<String, Double> map1,
      Map<String, Double> map2,
      FeatureVector featureVector) {
    for (Map.Entry<String, Double> a : map1.entrySet()) {
      for (Map.Entry<String, Double> b : map2.entrySet()) {
        Util.setDenseFeature(featureVector,
            a.getKey() + "^" + b.getKey(),
            Arrays.asList(a.getValue(), b.getValue()));
      }
    }
  }
}
