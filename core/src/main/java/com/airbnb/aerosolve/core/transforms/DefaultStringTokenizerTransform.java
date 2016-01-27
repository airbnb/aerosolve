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
 * "field2" specifies the regex used to tokenize
 * "field3" specifies whether field1 should be deleted or not
 */
public class DefaultStringTokenizerTransform extends Transform {
  private String fieldName1;
  private String regex;
  private boolean deleteField1;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    regex = config.getString(key + ".field2");
    deleteField1 = config.getBoolean(key + ".field3");
    outputName = config.getString(key + ".output");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    if (stringFeatures == null || floatFeatures == null) {
      return ;
    }

    Set<String> feature1 = stringFeatures.get(fieldName1);
    if (feature1 == null) {
      return;
    }

    Map<String, Double> output = Util.getOrCreateFloatFeature(outputName, floatFeatures);

    for (String rawString : feature1) {
      String[] tokenizedString = rawString.split(regex);
      for (String token : tokenizedString) {
        if (output.containsKey(token)) {
          double count = output.get(token);
          output.put(token, (count + 1.0));
        } else {
          output.put(token, 1.0);
        }
      }
    }

    if (deleteField1) {
      stringFeatures.remove(fieldName1);
    }
  }
}
