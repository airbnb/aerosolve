package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;

import java.util.Map;
import java.util.Set;

import com.typesafe.config.Config;

/**
 * Tokenizes and counts strings using a regex and optionally generates bigrams from the tokens
 * "field1" specifies the key of the feature
 * "regex" specifies the regex used to tokenize
 * "generateBigrams" specifies whether bigrams should also be generated
 */
public class DefaultStringTokenizerTransform implements Transform {
  public static final String BIGRAM_SEPARATOR = " ";

  private String fieldName1;
  private String regex;
  private String outputName;
  private boolean generateBigrams;
  private String bigramsOutputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    regex = config.getString(key + ".regex");
    outputName = config.getString(key + ".output");
    if (config.hasPath(key + ".generate_bigrams")) {
      generateBigrams = config.getBoolean(key + ".generate_bigrams");
    } else {
      generateBigrams = false;
    }
    if (generateBigrams) {
      bigramsOutputName = config.getString(key + ".bigrams_output");
    }
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

    Util.optionallyCreateFloatFeatures(featureVector);
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    Map<String, Double> output = Util.getOrCreateFloatFeature(outputName, floatFeatures);
    Map<String, Double> bigramOutput = null;
    if (generateBigrams) {
      bigramOutput = Util.getOrCreateFloatFeature(bigramsOutputName, floatFeatures);
    }

    for (String rawString : feature1) {
      if (rawString == null) continue;
      String[] tokenizedString = rawString.split(regex);
      for (String token : tokenizedString) {
        if (token.length() == 0) continue;
        incrementOutput(token, output);
      }
      if (generateBigrams) {
        String previousToken = null;
        for (String token : tokenizedString) {
          if (token.length() == 0) continue;
          if (previousToken == null) {
            previousToken = token;
          } else {
            String bigram = previousToken + BIGRAM_SEPARATOR + token;
            incrementOutput(bigram, bigramOutput);
            previousToken = token;
          }
        }
      }
    }
  }

  private static void incrementOutput(String key, Map<String, Double> output) {
    if (key == null || output == null) {
      return;
    }
    if (output.containsKey(key)) {
      double count = output.get(key);
      output.put(key, (count + 1.0));
    } else {
      output.put(key, 1.0);
    }
  }
}
