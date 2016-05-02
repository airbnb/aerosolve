package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.TransformUtil;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigObject;
import com.typesafe.config.ConfigValue;

import java.util.*;
import java.util.Map.Entry;

/**
 * A custom quantizer that quantizes features based on upper limits and bucket sizes from config
 * "field1" specifies feature family name.
 * If "select_features" is specified, we only transform features in the select_features list.
 * If "exclude_features" is specified, we transform features that are not in the exclude_features list.
 * If both "select_features" and "exclude_features" are specified, we transform features that are in
 * "select_features" list and not in "exclude_features" list.
 */
public class CustomLinearLogQuantizeTransform implements Transform {

  private String fieldName1;
  private String outputName;
  private TreeMap<Double, Double> limitBucketPairsMap;
  private double upperLimit;

  private List<String> excludeFeatures;
  private List<String> selectFeatures;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    outputName = config.getString(key + ".output");
    if (config.hasPath(key + ".exclude_features")) {
      excludeFeatures = config.getStringList(key + ".exclude_features");
    }

    if (config.hasPath(key + ".select_features")) {
      selectFeatures = config.getStringList(key + ".select_features");
    }
    limitBucketPairsMap =
        parseTokensOutOfLimitBucketPairs(config.getObjectList(key + ".limit_bucket"));
    upperLimit = limitBucketPairsMap.lastKey();
  }

  private static TreeMap<Double, Double> parseTokensOutOfLimitBucketPairs(
      List<? extends ConfigObject> pairs) {
    TreeMap<Double, Double> parsedTokensMap = new TreeMap<>();
    for (ConfigObject configObject : pairs) {
      List<Entry<String, ConfigValue>> entries = new ArrayList<>(configObject.entrySet());
      parsedTokensMap.put(Double.parseDouble(entries.get(0).getKey()),
                          Double.parseDouble(entries.get(0).getValue().unwrapped().toString()));
    }

    return parsedTokensMap;
  }

  private String transformFeature(String featureName,
                                  double featureValue,
                                  StringBuilder sb) {
    sb.setLength(0);
    sb.append(featureName);
    boolean isValueNegative = false;
    if (featureValue < 0.0) {
      isValueNegative = true;
      featureValue = -featureValue;
    }

    if (featureValue < 1e-2) {
      sb.append("=0.0");
    } else {
      double limit;
      double bucket;
      if (featureValue >= upperLimit) {
        featureValue = upperLimit;
        bucket = limitBucketPairsMap.get(upperLimit);
      } else {
        limit = limitBucketPairsMap.higherKey(featureValue);
        bucket = limitBucketPairsMap.get(limit);
      }

      Double val = TransformUtil.quantize(featureValue, bucket) * 1000;

      sb.append('=');
      if (isValueNegative) {
        sb.append('-');
      }

      sb.append(val.intValue()/1000.0);
    }

    return sb.toString();
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

    StringBuilder sb = new StringBuilder();
    for (Entry<String, Double> feature : feature1.entrySet()) {
      if ((excludeFeatures == null || !excludeFeatures.contains(feature.getKey())) &&
          (selectFeatures == null || selectFeatures.contains(feature.getKey()))) {
        String transformedFeature = transformFeature(feature.getKey(),
                                                     feature.getValue(),
                                                     sb);

        output.add(transformedFeature);
      }
    }
  }
}
