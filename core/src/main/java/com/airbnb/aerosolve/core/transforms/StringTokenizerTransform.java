package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;

import java.util.Collections;
import java.util.Map;
import java.util.Set;

import com.typesafe.config.Config;

/**
 * Tokenizes strings
 * "field1" specifies the key of the feature
 * "field2" specifies the delimiter or regex used to tokenize
 */
public class StringTokenizerTransform extends Transform {
  private String fieldName1;
  private String delimiterOrRegex;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    delimiterOrRegex = config.getString(key + ".field2");
    outputName = config.getString(key + ".output");
  }

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

    Set<String> output = Util.getOrCreateStringFeature(outputName, stringFeatures);

    for (String string : feature1) {
      String[] tokenizedString = string.split(delimiterOrRegex);
      Collections.addAll(output, tokenizedString);
    }
  }
}
