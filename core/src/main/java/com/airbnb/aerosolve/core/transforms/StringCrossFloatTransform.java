package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.features.Features;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.util.*;

/*
  if no prefix/keys provided, then cross whole family.
  prefix used in string, and keys used in float.
  to cross RAW features, use Features.RAW
 */
public class StringCrossFloatTransform implements Transform {
  private String fieldName1;
  // optional
  private Set<String> prefix;
  private String fieldName2;
  // optional
  private Set<String> keys;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    fieldName2 = config.getString(key + ".field2");
    outputName = config.getString(key + ".output");
    if (config.hasPath(key + ".prefix")) {
      prefix = new HashSet<>(config.getStringList(key + ".prefix"));
    }
    if (config.hasPath(key + ".keys")) {
      keys = new HashSet<>(config.getStringList(key + ".keys"));
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

    if (prefix != null) {
      Set<String> joint = new HashSet<>();
      for (String s1 : list1) {
        String p = Features.getStringFeatureName(s1);
        if (prefix.contains(p)) {
          joint.add(s1);
        }
      }
      if (joint.isEmpty()) return;
      list1 = joint;
    }

    if (keys != null) {
      Map<String, Double> joint = new HashMap<>();
      for (Map.Entry<String, Double> s2 : list2.entrySet()) {
        if (keys.contains(s2.getKey())) {
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
