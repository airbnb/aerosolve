package com.airbnb.aerosolve.core.transforms;

import com.typesafe.config.Config;

/**
 * Converts strings to either all lowercase or all uppercase
 * "field1" specifies the key of the feature
 * "convert_to_uppercase" converts strings to uppercase if true, otherwise converts to lowercase
 * "output" optionally specifies the key of the output feature, if it is not given the transform
 * overwrites / replaces the input feature
 */
public class ConvertStringCaseTransform extends StringTransform {
  private boolean convertToUppercase;

  @Override
  public void configure(Config config, String key) {
    super.configure(config, key);

    convertToUppercase = config.getBoolean(key + ".convert_to_uppercase");
  }

  @Override
  public String processString(String rawString) {
    if (rawString == null) {
      return null;
    }

    String convertedString;

    if (convertToUppercase) {
      convertedString = rawString.toUpperCase();
    } else {
      convertedString = rawString.toLowerCase();
    }

    return convertedString;
  }
}
