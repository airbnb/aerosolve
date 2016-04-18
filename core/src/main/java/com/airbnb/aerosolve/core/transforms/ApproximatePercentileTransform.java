package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.util.Map;

/**
 * Given a fieldName1, low, upper key
 * Remaps fieldName2's key2 value such that low = 0, upper = 1.0 thus approximating
 * the percentile using linear interpolation.
 */
public class ApproximatePercentileTransform implements Transform {
  private String fieldName1;
  private String fieldName2;
  private String lowKey;
  private String upperKey;
  private String key2;
  private String outputName;
  private String outputKey;
  private double minDiff;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    fieldName2 = config.getString(key + ".field2");
    lowKey = config.getString(key + ".low");
    upperKey = config.getString(key + ".upper");
    minDiff = config.getDouble(key + ".minDiff");
    key2 =  config.getString(key + ".key2");
    outputName = config.getString(key + ".output");
    outputKey = config.getString(key + ".outputKey");
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

    Map<String, Double> feature2 = floatFeatures.get(fieldName2);
    if (feature2 == null) {
      return;
    }

    Double val = feature2.get(key2);
    if (val == null) {
      return;
    }

    Double low = feature1.get(lowKey);
    Double upper = feature1.get(upperKey);
    
    if (low == null || upper == null) {
      return;
    }

    // Abstain if the percentiles are too close.
    double denom = upper - low;
    if (denom < minDiff) {
      return;
    }

    Map<String, Double> output = Util.getOrCreateFloatFeature(outputName, floatFeatures);

    Double outVal = 0.0;
    if (val <= low) {
      outVal = 0.0;
    } else if (val >= upper) {
      outVal = 1.0;
    } else {
      outVal = (val - low) / denom;
    }

    output.put(outputKey, outVal);
  }
}
