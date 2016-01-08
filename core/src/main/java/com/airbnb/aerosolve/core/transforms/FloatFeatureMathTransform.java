package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import com.google.common.base.Optional;
import java.util.function.DoubleFunction;

/**
 * Apply given Math function on specified float features defined by fieldName1 and keys
 * fieldName1: feature family name
 * keys: feature names
 * outputName: output feature family name (feature names or keys remain the same)
 * function: a string that specified the function that is going to apply to the given feature
 */
public class FloatFeatureMathTransform extends Transform {
  private String fieldName1; // feature family name
  private List<String> keys; // feature names
  private String outputName; // output feature family name
  private String functionName;   // a string that specified the function that is going to apply to the given feature
  private Optional<DoubleFunction<Double>> func;
  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    if (config.hasPath(key + ".keys")) {
      keys = config.getStringList(key + ".keys");
    }
    outputName = config.getString(key + ".output");
    functionName = config.getString(key + ".function");
    func = getFunction();
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    if (keys.isEmpty()) {
      return;
    }

    if (!func.isPresent()) {
      return;
    }

    if (floatFeatures == null) {
      return;
    }

    Map<String, Double> feature1 = floatFeatures.get(fieldName1);

    if (feature1 == null) {
      return;
    }
    Map<String, Double> output = new HashMap<>();
    for (String key : keys) {
      Double v = feature1.get(key);
      if (v != null) {
        output.put(key, func.get().apply(v));
      }
    }
    floatFeatures.put(outputName, output);
  }

  private Optional<DoubleFunction<Double>> getFunction() {
    switch (functionName) {
      case "sin":
        return Optional.of((double x) -> Math.sin(x));
      case "cos":
        return Optional.of((double x) -> Math.cos(x));
      case "log10":
        // return the original value if x <= 0
        return Optional.of((double x) -> Math.log10(x));
      case "log":
        // return the original value if x <= 0
        return Optional.of((double x) -> Math.log(x));
      case "abs":
        return Optional.of((double x) -> Math.abs(x));
    }
    return Optional.<DoubleFunction<Double>>absent();
  }
}
