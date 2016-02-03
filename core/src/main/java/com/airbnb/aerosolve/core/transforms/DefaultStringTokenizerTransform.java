package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;

import java.text.Normalizer;
import java.util.Collections;
import java.util.Map;
import java.util.Set;

import com.typesafe.config.Config;

/**
 * Tokenizes strings using a regex
 * "field1" specifies the key of the feature
 * "regex" specifies the regex used to tokenize
 */
public class DefaultStringTokenizerTransform extends Transform {
  private String fieldName1;
  private String regex;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    regex = config.getString(key + ".regex");
    outputName = config.getString(key + ".output");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    if (stringFeatures == null) {
      return ;
    }

    Set<String> feature1 = stringFeatures.get(fieldName1);
    if (feature1 == null) {
      return;
    }

    Util.optionallyCreateFloatFeatures(featureVector);
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    Map<String, Double> output = Util.getOrCreateFloatFeature(outputName, floatFeatures);

    for (String rawString : feature1) {
      String[] tokenizedString = rawString.split(regex);
      for (String token : tokenizedString) {
        if (token.length() == 0) continue;
        if (output.containsKey(token)) {
          double count = output.get(token);
          output.put(token, (count + 1.0));
        } else {
          output.put(token, 1.0);
        }
      }
    }
  }
}
