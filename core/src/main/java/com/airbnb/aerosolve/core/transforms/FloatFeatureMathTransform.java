package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
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

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    if (config.hasPath(key + ".keys")) {
      keys = config.getStringList(key + ".keys");
    }
    outputName = config.getString(key + ".output");
    functionName = config.getString(key + ".function");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    if (keys.isEmpty()) {
      return;
    }

    if (floatFeatures == null) {
      return;
    }
    Map<String, Double> feature1 = floatFeatures.get(fieldName1);

    if (feature1 == null) {
      return;
    }
    try {
      DoubleFunction<Double> func = getFunction();
      Map<String, Double> output = new HashMap<>();
      for (String key : keys) {
        Double v = feature1.get(key);
        if (v != null) {
          output.put(key, func.apply(v));
        }
      }
      floatFeatures.put(outputName, output);
    } catch(IllegalArgumentException e) {
      return;
    }
  }

  private DoubleFunction<Double> getFunction() {

    switch (functionName) {
      case "sin":
        return (double x) -> Math.sin(x);
      case "cos":
        return (double x) -> Math.cos(x);
      case "log10":
        // return the original value if x <= 0
        return (double x) -> x > 0 ? Math.log10(x) : x;
      case "log":
        // return the original value if x <= 0
        return (double x) -> x > 0 ? Math.log(x) : x;
      case "abs":
        return (double x) -> Math.abs(x);
    }
    throw new IllegalArgumentException("Function name is not recognized.");
  }
}
