package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.typesafe.config.Config;

/**
 * Tokenizes strings using a regex and generates ngrams from the tokens
 * "field1" specifies the key of the feature
 * "regex" specifies the regex used to tokenize
 * "n" specifies the size of the ngrams
 * "min_n" optional parameter, if specified ngrams from min_n (inclusive) to n (inclusive) will
 *  be generated and placed in the output
 */
public class NgramTransform implements Transform {
  public static final String BIGRAM_SEPARATOR = " ";

  private String fieldName1;
  private String regex;
  private int n;
  private int minN;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    regex = config.getString(key + ".regex");
    n = config.getInt(key + ".n");
    outputName = config.getString(key + ".output");
    if (config.hasPath(key + ".min_n")) {
      minN = config.getInt(key + ".min_n");
    } else {
      minN = n;
    }
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    if (regex == null) {
      return;
    }

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
      if (rawString == null) continue;

      String[] rawTokens = rawString.split(regex);
      ArrayList<String> cleanedTokens = new ArrayList<>();

      for (String token: rawTokens) {
        if (token != null && token.length() > 0) {
          cleanedTokens.add(token);
        }
      }

      for (int i = minN; i <= n; ++i) {
        List<String> ngrams = generateNgrams(cleanedTokens, i);

        for (String ngram : ngrams) {
          DefaultStringTokenizerTransform.incrementOutput(ngram, output);
        }
      }
    }
  }

  public static List<String> generateNgrams(ArrayList<String> tokens, int n) {
    List<String> ngrams = new LinkedList<>();

    if (n < 1 || tokens == null) {
      return ngrams;
    }

    for (int i = 0; i <= (tokens.size() - n); ++i) {
      ngrams.add(concatenate(tokens, i, (i + n)));
    }

    return ngrams;
  }

  private static String concatenate(ArrayList<String> tokens, int start, int end) {
    StringBuilder sb = new StringBuilder();

    for (int i = start; i < end; ++i) {
      String token = tokens.get(i);

      if (i > start) {
        sb.append(BIGRAM_SEPARATOR);
      }
      sb.append(token);
    }

    return sb.toString();
  }
}
