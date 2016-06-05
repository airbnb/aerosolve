package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.TransformUtil;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.util.Collection;
import java.util.Map;
import java.util.Set;

/**
 * Takes the floats in the keys of fieldName1 (or if keys are not specified, all floats) and
 * quantizes them into buckets. If the quantized float is less than or equal to a maximum specified
 * bucket value or greater than or equal to a minimum specified bucket value, then the quantized
 * float is stored as a string in a new string feature output specified by stringOutputName.
 * Otherwise, the original, unchanged float is stored in a new float feature output specified by
 * floatOutputName. The input float feature remains unchanged.
 */
public class MoveFloatToStringAndFloatTransform implements Transform {
  private String fieldName1;
  private Collection<String> keys;
  private double bucket;
  private double maxBucket;
  private double minBucket;
  private String stringOutputName;
  private String floatOutputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    if (config.hasPath(key + ".keys")) {
      keys = config.getStringList(key + ".keys");
    }
    bucket = config.getDouble(key + ".bucket");
    maxBucket = config.getDouble(key + ".max_bucket");
    minBucket = config.getDouble(key + ".min_bucket");
    stringOutputName = config.getString(key + ".string_output");
    floatOutputName = config.getString(key + ".float_output");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Map<String, Double>> floatFeatures = featureVector.floatFeatures;

    if (floatFeatures == null || floatFeatures.isEmpty()) {
      return;
    }

    Map<String, Double> input = floatFeatures.get(fieldName1);

    if (input == null || input.isEmpty()) {
      return;
    }

    Util.optionallyCreateStringFeatures(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    Set<String> stringOutput = Util.getOrCreateStringFeature(stringOutputName, stringFeatures);

    Map<String, Double> floatOutput = Util.getOrCreateFloatFeature(floatOutputName, floatFeatures);
    Collection<String> localKeys = (keys == null)? input.keySet() : keys;

    for (String key : localKeys) {
      moveFloatToStringAndFloat(
        input, key, bucket, minBucket, maxBucket, stringOutput, floatOutput);
    }
  }

  private static void moveFloatToStringAndFloat(
      Map<String, Double> input,
      String key,
      double bucket,
      double minBucket,
      double maxBucket,
      Set<String> stringOutput,
      Map<String, Double> floatOutput) {
    if (input.containsKey(key)) {
      Double inputFloatValue = input.get(key);

      Double inputFloatQuantized = TransformUtil.quantize(inputFloatValue, bucket);

      if (inputFloatQuantized >= minBucket && inputFloatQuantized <= maxBucket) {
        String movedFloat = key + "=" + inputFloatQuantized;
        stringOutput.add(movedFloat);
      } else {
        floatOutput.put(key, inputFloatValue);
      }
    }
  }
}
