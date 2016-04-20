package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;

import java.util.Map;
import java.util.Set;

import com.typesafe.config.Config;

/**
 * Takes the floats in fieldName1 and quantizes them into buckets. If the quantized float is
 * less than or equal to a maximum specified bucket value or greater than or equal to a minimum
 * specified bucket value, then the quantized float is stored as a string in a new string
 * feature output specified by stringOutputName. Otherwise, the original, unchanged float is
 * stored in a new float feature output specified by floatOutputName. The input float feature
 * remains unchanged.
 */
public class MoveFloatToStringAndFloatTransform implements Transform {
  private String fieldName1;
  private double bucket;
  private double maxBucket;
  private double minBucket;
  private String stringOutputName;
  private String floatOutputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
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

    for (Map.Entry<String, Double> inputEntry : input.entrySet()) {
      String inputFloatKey = inputEntry.getKey();
      Double inputFloatValue = inputEntry.getValue();

      Double inputFloatQuantized = LinearLogQuantizeTransform.quantize(inputFloatValue, bucket);

      if (inputFloatQuantized >= minBucket && inputFloatQuantized <= maxBucket) {
        String movedFloat = inputFloatKey + "=" + inputFloatQuantized;
        stringOutput.add(movedFloat);
      } else {
        floatOutput.put(inputFloatKey, inputFloatValue);
      }
    }
  }
}
