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
public class MultiscaleGridQuantizeTransform implements Transform {
  private String fieldName1;
  private List<Double> buckets;
  private String outputName;
  private String value1;
  private String value2;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    buckets = config.getDoubleList(key + ".buckets");
    outputName = config.getString(key + ".output");
    value1 = config.getString(key + ".value1");
    value2 = config.getString(key + ".value2");

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

    Double v1 = feature1.get(value1);
    Double v2 = feature1.get(value2);
    if (v1 == null || v2 == null) {
      return;
    }

    Util.optionallyCreateStringFeatures(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    Set<String> output = Util.getOrCreateStringFeature(outputName, stringFeatures);
    transformFeature(v1, v2, buckets, output);
  }

  public static void transformFeature(double v1, double v2, List<Double> buckets, Set<String> output) {
    for (Double bucket : buckets) {
      transformFeature(v1, v2, bucket, output);
    }
  }

  public static void transformFeature(double v1, double v2, double bucket, Set<String> output) {
    double q1 = LinearLogQuantizeTransform.quantize(v1, bucket);
    double q2 = LinearLogQuantizeTransform.quantize(v2, bucket);
    output.add("[" + bucket + "]=(" + q1 + ',' + q2 + ')');
  }
}
