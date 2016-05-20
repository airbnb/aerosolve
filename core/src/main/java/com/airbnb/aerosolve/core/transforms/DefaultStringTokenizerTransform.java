package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.util.Map;
import java.util.Set;

/**
 * Tokenizes and counts strings using a regex and optionally generates bigrams from the tokens
 * "field1" specifies the key of the feature
 * "regex" specifies the regex used to tokenize
 * "generateBigrams" specifies whether bigrams should also be generated
 */
public class DefaultStringTokenizerTransform implements Transform {
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

    NgramTransform.generateOutputTokens(feature1, regex, output, 1, 1);

    if (generateBigrams) {
      bigramOutput = Util.getOrCreateFloatFeature(bigramsOutputName, floatFeatures);

      NgramTransform.generateOutputTokens(feature1, regex, bigramOutput, 2, 2);
    }
  }
}
