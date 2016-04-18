package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.util.*;
import java.util.Map.Entry;

/**
 * Created by hector_yee on 8/25/14.
 * Quantizes the floatFeature named in "field1" with buckets in "bucket" before placing
 * it in the stringFeature named "output"
 */
public class MultiscaleQuantizeTransform implements Transform {
  private String fieldName1;
  private List<Double> buckets;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    buckets = config.getDoubleList(key + ".buckets");
    outputName = config.getString(key + ".output");
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
      transformAndAddFeature(buckets,
                             feature.getKey(),
                             feature.getValue(),
                             output);
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
