package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;
import java.util.Set;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;

/**
 * Created by hector_yee on 8/25/14.
 * Multiplies the floatFeature named in "field1" with "scale" before placing
 * it in the stringFeature named "output"
 */
public class QuantizeTransform implements Transform {
  private String fieldName1;
  private double scale;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    scale = config.getDouble(key + ".scale");
    outputName = config.getString(key + ".output");
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
      transformAndAddFeature(scale,
                             feature.getKey(),
                             feature.getValue(),
                             output);
    }
  }

  public static void transformAndAddFeature(Double scale,
                                     String featureName,
                                     Double featureValue,
                                     Set<String> output) {
    Double dbl = featureValue * scale;
    int val = dbl.intValue();
    output.add(featureName + '=' + val);
  }
}
