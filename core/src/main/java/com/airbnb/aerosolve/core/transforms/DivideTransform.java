package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;

import com.typesafe.config.Config;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**
 * output = field1.keys / (field2.key2 + constant)
 * If keys are provided, features specified in keys from field1 are considered, otherwise
 * all features in field1 are considered
 */
public class DivideTransform extends Transform {
  private String fieldName1;
  private String fieldName2;
  private List<String> keys;
  private String key2;
  private String outputName;
  private Double constant;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    fieldName2 = config.getString(key + ".field2");
    if (config.hasPath(key + ".keys")) {
      keys = config.getStringList(key + ".keys");
    }
    key2 = config.getString(key + ".key2");
    constant = config.getDouble((key + ".constant"));
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
    Double div = feature2.get(key2);
    if (div == null) {
      return;
    }

    Double scale = 1.0 / (constant + div);
    Map<String, Double> output = Util.getOrCreateFloatFeature(outputName, floatFeatures);

    for (Entry<String, Double> f1 : feature1.entrySet()) {
      String key = f1.getKey();
      if (keys == null || keys.contains(key)) {
        Double val = f1.getValue();
        if (val != null) {
          output.put(key + "-d-" + key2, val * scale);
        }
      }
    }
  }
}
