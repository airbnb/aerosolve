package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.transforms.types.FloatTransform;

import java.util.List;
import java.util.Map;

import com.typesafe.config.Config;

/**
 * Fill in a default value for float features when features are missing
 */
public class CoalesceFloatTransform extends FloatTransform {
  private List<String> keys;
  private double value;

  @Override
  public void init(Config config, String key) {
    keys = config.getStringList(key + ".keys");
    value = config.getDouble(key + ".value");
  }

  @Override
  public void output(Map<String, Double> input, Map<String, Double> output) {
    for (String key : keys) {
      Double v = input.get(key);
      if (v == null) {
        output.put(key, value);
      }
    }
  }
}
