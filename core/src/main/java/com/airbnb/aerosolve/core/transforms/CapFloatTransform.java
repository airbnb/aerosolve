package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.transforms.types.FloatTransform;
import com.typesafe.config.Config;

import java.util.List;
import java.util.Map;

public class CapFloatTransform extends FloatTransform {
  private List<String> keys;
  private double lowerBound;
  private double upperBound;

  @Override
  public void init(Config config, String key) {
    keys = config.getStringList(key + ".keys");
    lowerBound = config.getDouble(key + ".lower_bound");
    upperBound = config.getDouble(key + ".upper_bound");
  }

  @Override
  public void output(Map<String, Double> input, Map<String, Double> output) {
    for (String key : keys) {
      Double v = input.get(key);
      if (v != null) {
        output.put(key, Math.min(upperBound, Math.max(lowerBound, v)));
      }
    }
  }
}
