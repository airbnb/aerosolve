package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;

import java.util.Map;
import java.util.Set;

import com.typesafe.config.Config;

/**
 * Replaces all substrings that match a given regex with a replacement string
 * "field1" specifies the key of the feature
 * "regex" specifies the regex used to perform replacements
 * "replacement" specifies the replacement string
 */
public class ReplaceAllStringsTransform extends Transform {
  private String fieldName1;
  private String regex;
  private String replacement;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
//    TODO (christhetree): is it possible to get a map of regexes and replacements instead?
    regex = config.getString(key + ".regex");
    replacement = config.getString(key + ".replacement");
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

    for (String rawString : feature1) {
      if (rawString == null) continue;
      String processedString = rawString.replaceAll(regex, replacement);
      output.add(processedString);
    }
  }
}
