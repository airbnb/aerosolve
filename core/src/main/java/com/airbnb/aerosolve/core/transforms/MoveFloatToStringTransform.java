package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.TransformUtil;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.util.Iterator;
import java.util.Set;
import java.util.Map;
import java.util.Map.Entry;
import java.util.List;

/**
 * Moves named fields from one family to another. If keys are not specified, all keys are moved
 * from the float family. Features are capped via a `cap` config, which defaults to 1e10, to avoid
 * exploding string features. The original float feature is removed but can be overridden using
 * `keep` boolean config.
 */
public class MoveFloatToStringTransform implements Transform {
  private String fieldName1;
  private double bucket;
  private String outputName;
  private List<String> keys;
  private double cap;
  private boolean keep;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    bucket = config.getDouble(key + ".bucket");
    outputName = config.getString(key + ".output");
    if (config.hasPath(key + ".keys")) {
      keys = config.getStringList(key + ".keys");
    }
    if (config.hasPath(key + ".cap")) {
      cap = config.getDouble(key + ".cap");
    } else {
      cap = 1e10;
    }
    if (config.hasPath(key + ".keep")) {
      keep = config.getBoolean(key + ".keep");
    } else {
      keep = false;
    }
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

    if (keys != null) {
      for (String key : keys) {
        moveFloat(feature1, output, key, cap, bucket);
        if(!keep) {
          feature1.remove(key);
        }
      }
    } else {
      for (Iterator<Entry<String, Double>> iterator = feature1.entrySet().iterator();
          iterator.hasNext();) {
        Entry<String, Double> entry = iterator.next();
        String key = entry.getKey();

        moveFloat(feature1, output, key, cap, bucket);
        if(!keep) {
          iterator.remove();
        }
      }
    }
  }

  public static void moveFloat(
      Map<String, Double> feature1,
      Set<String> output,
      String key,
      double cap,
      double bucket) {
    if (feature1.containsKey(key)) {
      Double dbl = feature1.get(key);
      if (dbl > cap) {
        dbl = cap;
      }

      Double quantized = TransformUtil.quantize(dbl, bucket);
      output.add(key + '=' + quantized);
    }
  }
}
