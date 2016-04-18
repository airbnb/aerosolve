package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.util.Map;
import java.util.Set;

/**
 * Converts strings to either all lowercase or all uppercase
 * "field1" specifies the key of the feature
 * "convert_to_uppercase" converts strings to uppercase if true, otherwise converts to lowercase
 */
public class ConvertStringCaseTransform implements Transform {
  private String fieldName1;
  private boolean convertToUppercase;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    convertToUppercase = config.getBoolean(key + ".convert_to_uppercase");
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
      String convertedString;
      if (convertToUppercase) {
        convertedString = rawString.toUpperCase();
      } else {
        convertedString = rawString.toLowerCase();
      }
      output.add(convertedString);
    }
  }
}
