package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.util.*;
import java.util.Map.Entry;

/**
 * Quantizes the floatFeature named in "field1" with buckets in "bucket" before placing
 * it in the floatFeature named "output" subtracting the origin of the box.
 */
public class MultiscaleGridContinuousTransform implements Transform {
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

    Map<String, Double> output = Util.getOrCreateFloatFeature(outputName, floatFeatures);

    transformFeature(v1, v2, output);
  }

  public void transformFeature(double v1,
                               double v2,
                               Map<String, Double> output) {
    for (Double bucket : buckets) {
      transformFeature(v1, v2, bucket, output);
    }
  }

  public static void transformFeature(double v1,
                                      double v2,
                                      double bucket,
                                      Map<String, Double> output) {
    Double mult1 = v1 / bucket;
    double q1 = bucket * mult1.intValue();
    Double mult2 = v2 / bucket;
    double q2 = bucket * mult2.intValue();
    String bucketName = "[" + bucket + "]=(" + q1 + ',' + q2 + ')';
    output.put(bucketName + "@1", v1 - q1);
    output.put(bucketName + "@2", v2 - q2);
  }
}
