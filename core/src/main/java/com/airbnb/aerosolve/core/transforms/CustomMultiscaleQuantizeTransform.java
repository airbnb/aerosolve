package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.util.*;
import java.util.Map.Entry;

/**
 * Quantize the floatFeature named in "field1" with buckets in "bucket" before placing
 * it in the stringFeature named "output".
 * "field1" specifies feature family name.
 * If "select_features" is specified, we only transform features in the select_features list.
 * If "exclude_features" is specified, we transform features that are not in the exclude_features list.
 * If both "select_features" and "exclude_features" are specified, we transform features that are in
 * "select_features" list and not in "exclude_features" list.
 */

public class CustomMultiscaleQuantizeTransform implements Transform {
  private String fieldName1;
  private List<Double> buckets;
  private String outputName;
  private List<String> excludeFeatures;
  private List<String> selectFeatures;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    buckets = config.getDoubleList(key + ".buckets");
    outputName = config.getString(key + ".output");
    if (config.hasPath(key + ".exclude_features")) {
      excludeFeatures = config.getStringList(key + ".exclude_features");
    }

    if (config.hasPath(key + ".select_features")) {
      selectFeatures = config.getStringList(key + ".select_features");
    }
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

    Util.optionallyCreateStringFeatures(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    Set<String> output = Util.getOrCreateStringFeature(outputName, stringFeatures);

    for (Entry<String, Double> feature : feature1.entrySet()) {
      if ((excludeFeatures == null || !excludeFeatures.contains(feature.getKey())) &&
          (selectFeatures == null || selectFeatures.contains(feature.getKey()))) {
        transformAndAddFeature(buckets,
            feature.getKey(),
            feature.getValue(),
            output);
      }
    }
  }

  public static void transformAndAddFeature(List<Double> buckets,
                                            String featureName,
                                            Double featureValue,
                                            Set<String> output) {
    if (featureValue == 0.0) {
      output.add(featureName + "=0");
      return;
    }

    for (double bucket : buckets) {
      double quantized = LinearLogQuantizeTransform.quantize(featureValue, bucket);
      output.add(featureName + '[' + bucket + "]=" + quantized);
    }
  }
}
