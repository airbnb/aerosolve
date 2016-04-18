package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;

/**
 * A quantizer that starts out with linearly space buckets that get coarser and coarser
 * and eventually transitions to log buckets.
 */
public class LinearLogQuantizeTransform implements Transform {
  private String fieldName1;
  private String outputName;

  private static StringBuilder sb;
  // Upper limit of each bucket to check if feature value falls in the bucket
  private static List<Double> limits;
  // Step size used for quantization, for the correponding limit
  private static List<Double> stepSizes;
  // Limit beyond which quantized value would be rounded to integer (ignoring decimals)
  private static double integerRoundingLimit;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    outputName = config.getString(key + ".output");
    sb = new StringBuilder();

    limits = new ArrayList<>();
    stepSizes = new ArrayList<>();

    limits.add(1.0);
    stepSizes.add(1.0 / 32.0);

    limits.add(10.0);
    stepSizes.add(0.125);

    limits.add(25.0);
    stepSizes.add(0.25);

    limits.add(50.0);
    stepSizes.add(5.0);

    limits.add(100.0);
    stepSizes.add(10.0);

    limits.add(400.0);
    stepSizes.add(25.0);

    limits.add(2000.0);
    stepSizes.add(100.0);

    limits.add(10000.0);
    stepSizes.add(250.0);

    integerRoundingLimit = 25.0;
  }

  private static boolean checkAndQuantize(Double featureValue, double limit, double stepSize, boolean integerRounding) {
    if (featureValue <= limit) {
      if (!integerRounding) {
        sb.append(quantize(featureValue, stepSize));
      } else {
        sb.append(quantize(featureValue, stepSize).intValue());
      }

      return true;
    }

    return false;
  }

  private static String logQuantize(String featureName, double featureValue) {
    sb.setLength(0);
    sb.append(featureName);
    sb.append('=');

    Double dbl = featureValue;
    if (dbl < 0.0) {
      sb.append('-');
      dbl = -dbl;
    }
    // At every stage we quantize roughly to a precision 10% of the magnitude.
    if (dbl < 1e-2) {
      sb.append('0');
    } else {
      boolean isQuantized = false;
      for (int i = 0; i < limits.size(); i++) {
        Double limit = limits.get(i);
        Double stepSize = stepSizes.get(i);
        if (limit > integerRoundingLimit) {
          isQuantized = checkAndQuantize(dbl, limit, stepSize, true);
        } else {
          isQuantized = checkAndQuantize(dbl, limit, stepSize, false);
        }

        if (isQuantized) {
          break;
        }
      }

      if (! isQuantized) {
        Double exp = Math.log(dbl) / Math.log(2.0);
        Long val = 1L << exp.intValue();
        sb.append(val);
      }
    }

    return sb.toString();
  }

  public static Double quantize(double val, double delta) {
    Double mult = val / delta;
    return delta * mult.intValue();
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    if (floatFeatures == null) {
      return;
    }
    Map<String, Double> feature1 = floatFeatures.get(fieldName1);
    if (feature1 == null || feature1.isEmpty()) {
      return;
    }

    Util.optionallyCreateStringFeatures(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    Set<String> output = Util.getOrCreateStringFeature(outputName, stringFeatures);

    for (Entry<String, Double> feature : feature1.entrySet()) {
      output.add(logQuantize(feature.getKey(), feature.getValue()));
    }
  }
}
