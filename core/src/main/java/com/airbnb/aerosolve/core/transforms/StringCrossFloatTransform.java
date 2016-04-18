package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;

import com.typesafe.config.Config;
import java.util.HashMap;
import java.util.Set;
import java.util.Map;

public class StringCrossFloatTransform implements Transform {
  private String fieldName1;
  private String fieldName2;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    fieldName2 = config.getString(key + ".field2");
    outputName = config.getString(key + ".output");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Set<String>> stringFeatures = featureVector.stringFeatures;
    Map<String, Map<String, Double>> floatFeatures = featureVector.floatFeatures;
    if (stringFeatures == null || stringFeatures.isEmpty()) return;
    if (floatFeatures == null || floatFeatures.isEmpty()) return;

    Set<String> list1 = stringFeatures.get(fieldName1);
    if (list1 == null || list1.isEmpty()) return;
    Map<String, Double> list2 = floatFeatures.get(fieldName2);
    if (list2 == null || list2.isEmpty()) return;

    Map<String, Double> output = Util.getOrCreateFloatFeature(outputName, floatFeatures);

    for (String s1 : list1) {
      for (Map.Entry<String, Double> s2 : list2.entrySet()) {
        output.put(s1 + "^" + s2.getKey(), s2.getValue());
      }
    }
  }
}
