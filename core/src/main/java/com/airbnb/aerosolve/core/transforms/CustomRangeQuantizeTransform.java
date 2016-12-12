package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;

import java.util.List;
import java.util.Map;
import java.util.Set;

import com.typesafe.config.Config;

/**
 * A custom quantizer that quantizes features based on the specified range in the config. "field1":
 * specifies feature family name If "select_features" is specified, we only transform features in
 * the select_features list, otherwise, we transform all features in the feature family thresholds:
 * specifies how do we quantize the feature, for example: if we set thresholds: [0.1, 0.5] the
 * transformer will bucketize features into three buckets: the first with feature value <= 0.1, the
 * second with features value > 0.1 and <=0.5; and the third with value > 0.5. The quantized
 * features are put under a new feature family specified by "output"
 * Note that the transformer assumes thresholds are in ascending order
 */
public class CustomRangeQuantizeTransform implements Transform {
  private String fieldName1;
  private List<Double> thresholds;
  private String outputName;
  private List<String> selectFeatures;

  private static void getQuantizedFeatures(List<Double> thresholds,
                                           String featureName,
                                           Double featureValue,
                                           Set<String> output) {

    double tMin = thresholds.get(0);
    double tMax = thresholds.get(thresholds.size() - 1);
    if (featureValue <= tMin) {
      output.add(featureName + "<=" + Double.toString(tMin));
      return;
    }

    if (featureValue > tMax) {
      output.add(featureName + ">" + Double.toString(tMax));
      return;
    }

    for (int i = 0; i < thresholds.size() - 1; i++) {
      double t0 = thresholds.get(i);
      double t1 = thresholds.get(i + 1);
      if (featureValue > t0 && featureValue <= t1) {
        output.add(Double.toString(t0) + "<" + featureName + "<=" + Double.toString(t1));
        return;
      }
    }
  }

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    thresholds = config.getDoubleList(key + ".thresholds");
    outputName = config.getString(key + ".output");

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

    for (Map.Entry<String, Double> feature : feature1.entrySet()) {
      if ((selectFeatures == null || selectFeatures.contains(feature.getKey()))) {
        getQuantizedFeatures(thresholds,
            feature.getKey(),
            feature.getValue(),
            output);
      }
    }
  }
}
