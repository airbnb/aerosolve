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
 * Buckets float features and places them in a new float column.
 */
public class BucketFloatTransform implements Transform {
  private String fieldName1;
  private double bucket;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    bucket = config.getDouble(key + ".bucket");
    outputName = config.getString(key + ".output");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();

    if (floatFeatures == null) {
      return;
    }
    Map<String, Double> feature1 = floatFeatures.get(fieldName1);
    if (feature1 == null || feature1.isEmpty()) {
      return;
    }

    Map<String, Double> output = Util.getOrCreateFloatFeature(outputName, floatFeatures);

    for (Entry<String, Double> feature : feature1.entrySet()) {
      Double dbl = LinearLogQuantizeTransform.quantize(feature.getValue(), bucket);
      Double newVal = feature.getValue() - dbl;
      String name = feature.getKey() + '[' + bucket + "]=" + dbl;
      output.put(name, newVal);
    }
  }
}
