package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.TransformUtil;
import com.airbnb.aerosolve.core.util.Util;

import java.util.Map;

import com.typesafe.config.Config;

/**
 * Takes the floats in fieldName1, quantizes them into buckets, converts them to strings, then
 * crosses them with the floats in fieldName2 and then stores the result in a new float feature
 * output specified by outputName.
 */
public class FloatCrossFloatTransform implements Transform {
  private String fieldName1;
  private double bucket;
  private double cap;
  private String fieldName2;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    bucket = config.getDouble(key + ".bucket");
    if (config.hasPath(key + ".cap")) {
      cap = config.getDouble(key + ".cap");
    } else {
      cap = 1e10;
    }
    fieldName2 = config.getString(key + ".field2");
    outputName = config.getString(key + ".output");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Map<String, Double>> floatFeatures = featureVector.floatFeatures;

    if (floatFeatures == null || floatFeatures.isEmpty()) {
      return;
    }

    Map<String, Double> map1 = floatFeatures.get(fieldName1);

    if (map1 == null || map1.isEmpty()) {
      return;
    }

    Map<String, Double> map2 = floatFeatures.get(fieldName2);

    if (map2 == null || map2.isEmpty()) {
      return;
    }

    Map<String, Double> output = Util.getOrCreateFloatFeature(outputName, floatFeatures);

    for (Map.Entry<String, Double> entry1 : map1.entrySet()) {
      String float1Key = entry1.getKey();
      Double float1Value = entry1.getValue();

      if (float1Value > cap) {
        float1Value = cap;
      }

      Double float1Quantized = TransformUtil.quantize(float1Value, bucket);

      for (Map.Entry<String, Double> entry2 : map2.entrySet()) {
        String float2Key = entry2.getKey();
        Double float2Value = entry2.getValue();

        String outputKey = float1Key + "=" + float1Quantized + "^" + float2Key;

        output.put(outputKey, float2Value);
      }
    }
  }
}
