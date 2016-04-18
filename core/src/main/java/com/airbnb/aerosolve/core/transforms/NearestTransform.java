package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.util.HashMap;
import java.util.Set;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;

/**
 * output = nearest of (field1, field2.key)
 */
public class NearestTransform implements Transform {
  private String fieldName1;
  private String fieldName2;
  private String key2;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    fieldName2 = config.getString(key + ".field2");
    key2 = config.getString(key + ".key");
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
    
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    Set<String> output = Util.getOrCreateStringFeature(outputName, stringFeatures);
    String nearest = "nothing";
    double bestDist = 1e10;

    for (Entry<String, Double> f1 : feature1.entrySet()) {
      double dist = Math.abs(f1.getValue() - sub);
      if (dist < bestDist) {
        nearest = f1.getKey();
        bestDist = dist;
      }
    }
    output.add(key2 + "~=" + nearest);

    Util.optionallyCreateStringFeatures(featureVector);
  }
}
