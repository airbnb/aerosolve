package com.airbnb.aerosolve.core.transforms.types;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.transforms.Transform;
import com.airbnb.aerosolve.core.util.Util;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.typesafe.config.Config;

/**
 * Abstract representation of a transform that processes all strings in a string feature and
 * outputs a new string feature or overwrites /replaces the input string feature.
 * "field1" specifies the key of the feature
 * "output" optionally specifies the key of the output feature, if it is not given the transform
 * overwrites / replaces the input feature
 */
public abstract class StringTransform implements Transform {
  protected String fieldName1;
  protected String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    if (config.hasPath(key + ".output")) {
      outputName = config.getString(key + ".output");
    } else {
      outputName = fieldName1;
    }
    init(config, key);
  }

  protected abstract void init(Config config, String key);

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    if (stringFeatures == null) {
      return;
    }

    Set<String> feature1 = stringFeatures.get(fieldName1);
    if (feature1 == null) {
      return;
    }

    HashSet<String> processedStrings = new HashSet<>();

    for (String rawString : feature1) {
      if (rawString != null) {
        String processedString = processString(rawString);
        processedStrings.add(processedString);
      }
    }

    Set<String> output = Util.getOrCreateStringFeature(outputName, stringFeatures);

    // Check reference equality to determine whether the output should overwrite the input
    if (output == feature1) {
      output.clear();
    }

    output.addAll(processedStrings);
  }

  public abstract String processString(String rawString);
}
