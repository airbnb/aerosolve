package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.features.Features;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.util.*;

/*
  if no keys1/keys2 provided, then cross whole family.
  otherwise cross features.
  keys1 is string features, and keys2 is float features.
  to cross RAW string features, put Features.RAW in keys1
 */
public class StringCrossFloatTransform implements Transform {
  private String fieldName1;
  // optional
  private Set<String> keys1;
  private String fieldName2;
  // optional
  private Set<String> keys2;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    fieldName2 = config.getString(key + ".field2");
    outputName = config.getString(key + ".output");
    if (config.hasPath(key + ".keys1")) {
      keys1 = new HashSet<>(config.getStringList(key + ".keys1"));
    }
    if (config.hasPath(key + ".keys2")) {
      keys2 = new HashSet<>(config.getStringList(key + ".keys2"));
    }
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

    if (keys1 != null) {
      Set<String> joint = new HashSet<>();
      for (String s1 : list1) {
        String p = Features.getStringFeatureName(s1);
        if (keys1.contains(p)) {
          joint.add(s1);
        }
      }
      if (joint.isEmpty()) return;
      list1 = joint;
    }

    if (keys2 != null) {
      Map<String, Double> joint = new HashMap<>();
      for (Map.Entry<String, Double> s2 : list2.entrySet()) {
        if (keys2.contains(s2.getKey())) {
          joint.put(s2.getKey(), s2.getValue());
        }
      }
      if (joint.isEmpty()) return;
      list2 = joint;
    }

    Map<String, Double> output = Util.getOrCreateFloatFeature(outputName, floatFeatures);

    for (String s1 : list1) {
      for (Map.Entry<String, Double> s2 : list2.entrySet()) {
        output.put(s1 + "^" + s2.getKey(), s2.getValue());
      }
    }
  }
}
