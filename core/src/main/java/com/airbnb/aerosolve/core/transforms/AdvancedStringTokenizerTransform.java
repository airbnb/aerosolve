package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;

import java.text.Normalizer;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.ImmutableMap;
import com.typesafe.config.Config;

/**
 * Tokenizes strings using a series of regexes and pre-processing
 * "field1" specifies the key of the feature
 * "regex" specifies the regex used to tokenize
 */
public class AdvancedStringTokenizerTransform extends Transform {
  public static final Map<String, String> NORMALIZE_QUOTES_REPLACEMENT_MAP =
      ImmutableMap.<String, String>builder()
          // TODO: add all (regex, replacement) pairs
          .put("regex", "replacement")
          .build();

  public static final Map<String, String> NORMALIZE_CONTRACTIONS_REPLACEMENT_MAP =
      ImmutableMap.<String, String>builder()
          // TODO: add all (regex, replacement) pairs
          .put("regex", "replacement")
          .build();

  public static final Map<String, String> NORMALIZE_TOKENIZER_REPLACEMENT_MAP =
      ImmutableMap.<String, String>builder()
          // TODO: add all (regex, replacement) pairs
          .put("regex", "replacement")
          .build();

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
      String normalizedString = normalizeContractions(normalizeQuotes(normalizeUtf8(rawString)));
      String[] tokenizedString = tokenizer(normalizedString, regex);

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

  private static String normalizeUtf8(String rawString) {
    if (rawString == null) {
      return null;
    }
    return Normalizer.normalize(rawString, Normalizer.Form.NFC);
  }

  private static String normalizeQuotes(String rawString) {
    if (rawString == null) {
      return null;
    }
    return replace(rawString, NORMALIZE_QUOTES_REPLACEMENT_MAP);
  }

  private static String normalizeContractions(String rawString) {
    if (rawString == null) {
      return null;
    }
    return replace(rawString, NORMALIZE_CONTRACTIONS_REPLACEMENT_MAP);
  }

  private static String[] tokenizer(String rawString, String regex) {
    if (rawString == null) {
      return null;
    }
    rawString = replace(rawString, NORMALIZE_TOKENIZER_REPLACEMENT_MAP);
    return rawString.split(regex);
  }

  private static String replace(String rawString, Map<String, String> replacementMap) {
    if (rawString == null) {
      return null;
    }
    if (replacementMap == null) {
      return rawString;
    }
    for (Map.Entry<String, String> replacementEntry : replacementMap.entrySet()) {
      String regex = replacementEntry.getKey();
      String replacement = replacementEntry.getValue();
      rawString = rawString.replaceAll(regex, replacement);
    }
    return rawString;
  }
}
