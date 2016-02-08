package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;

import java.text.Normalizer;
import java.util.Map;
import java.util.Set;

import com.typesafe.config.Config;

/**
 * Normalizes strings to the UTF-8 NFC standard
 * "field1" specifies the key of the feature
 */
public class NormalizeUtf8Transform extends Transform {
  private String fieldName1;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
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
      String normalizedString = Normalizer.normalize(rawString, Normalizer.Form.NFC);
      output.add(normalizedString);
    }
  }
}
