package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;
import lombok.extern.slf4j.Slf4j;

import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Similar to MoveFloatToStringTransform, however, just move defined float value into String Feature
 * not using bucket. This is used when there are certain number of incorrect data,
 * i.e. x = 0 doesn't mean it is worse than x = 0.00001, it just somewhere in the pipeline
 * make null = 0, so before we fixed the pipeline, convert it to string feature.
 */
@Slf4j
public class FloatToStringTransform implements Transform {
  private String fieldName;
  private Collection<String> keys;
  private Set<Double> values;
  private String stringOutputName;

  @Override
  public void configure(Config config, String key) {
    fieldName = config.getString(key + ".field1");
    if (config.hasPath(key + ".keys")) {
      keys = config.getStringList(key + ".keys");
    }
    values = new HashSet<>(config.getDoubleList(key + ".values"));
    stringOutputName = config.getString(key + ".string_output");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Map<String, Double>> floatFeatures = featureVector.floatFeatures;

    if (floatFeatures == null || floatFeatures.isEmpty()) {
      return;
    }

    Map<String, Double> input = floatFeatures.get(fieldName);

    if (input == null || input.isEmpty()) {
      return;
    }

    Util.optionallyCreateStringFeatures(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    Set<String> stringOutput = Util.getOrCreateStringFeature(stringOutputName, stringFeatures);
    Collection<String> localKeys = (keys == null)? input.keySet() : keys;
    log.debug("k {} {}", localKeys, input);
    for (String key : localKeys) {
      moveFloatToString(
          input, key, values, stringOutput);
    }
  }

  private void moveFloatToString(
      Map<String, Double> input,
      String key, Set<Double> values,
      Set<String> stringOutput) {
    if (input.containsKey(key)) {
      Double inputFloatValue = input.get(key);

      if (values.contains(inputFloatValue)) {
        String movedFloat = key + "=" + inputFloatValue;
        stringOutput.add(movedFloat);
        input.remove(key);
      }
    }
  }
}
