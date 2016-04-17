package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;

import com.typesafe.config.Config;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.HashMap;

/**
 * Computes the polynomial product of all values in field1
 * i.e. prod_i 1 + x_i
 * and places the result in outputName
 */
public class ProductTransform implements Transform {
  private String fieldName1;
  private List<String> keys;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    outputName = config.getString(key + ".output");
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

    Double prod = 1.0;
    Map<String, Double> output = Util.getOrCreateFloatFeature(outputName, floatFeatures);
    for (String key : keys) {
      Double dbl = feature1.get(key);
      if (dbl != null) {
        prod *= 1.0 + dbl;
      }
    }
    output.put("*", prod);
  }
}
