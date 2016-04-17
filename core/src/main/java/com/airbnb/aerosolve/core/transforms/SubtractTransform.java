package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;

import com.typesafe.config.Config;

import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;

/**
 * output = field1 - field2.key
 */
public class SubtractTransform implements Transform {
  private String fieldName1;
  private String fieldName2;
  private List<String> keys;
  private String key2;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    fieldName2 = config.getString(key + ".field2");
    keys = config.getStringList(key + ".keys");
    key2 = config.getString(key + ".key2");
    outputName = config.getString(key + ".output");
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

    Map<String, Double> feature2 = floatFeatures.get(fieldName2);
    if (feature2 == null) {
      return;
    }
    Double sub = feature2.get(key2);
    if (sub == null) {
      return;
    }

    Map<String, Double> output = Util.getOrCreateFloatFeature(outputName, floatFeatures);

    for (String key : keys) {
      if (feature1.containsKey(key)) {
        Double val = feature1.get(key);
        if (val != null) {
          output.put(key + '-' + key2, val - sub);
        }
      }
    }
  }
}
