package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.util.*;

/**
 * A transform that applies the winner takes all hash to
 * A set of dense feature families and emits tokens to a string feature family.
 * See "The power of comparative reasoning"
 * http://research.google.com/pubs/pub37298.html
 * For ease of use we use the recommended window size of 4 features
 * to generate 2-bit tokens
 * and pack each word with num_tokens_per_word of these.
 */
public class WtaTransform implements Transform {
  private List<String> fieldNames;
  private String outputName;
  private int seed;
  private int numWordsPerFeature;
  private int numTokensPerWord;
  private final byte windowSize = 4;

  @Override
  public void configure(Config config, String key) {
    // What fields to use to construct the hash.
    fieldNames = config.getStringList(key + ".field_names");
    // Name of field to output to.
    outputName = config.getString(key + ".output");
    // The seed of the random number generator.
    seed = config.getInt(key + ".seed");
    // The number of words per feature.
    numWordsPerFeature = config.getInt(key + ".num_words_per_feature");
    // The number of tokens per word.
    numTokensPerWord = config.getInt(key + ".num_tokens_per_word");
    assert(numTokensPerWord <= 16);
  }

  // Generates a permutation of the array and appends it
  // to a given deque.
  private void generatePermutation(int size,
                                   Random rnd,
                                   Deque<Integer> dq) {
    dq.clear();
    int[] permutation = new int[size];
    for (int i = 0; i < size; i++) {
      permutation[i] = i;
    }
    for (int i = 0; i < size; i++) {
      int other = rnd.nextInt(size);
      int tmp = permutation[i];
      permutation[i] = permutation[other];
      permutation[other] = tmp;
    }
    for (int i = 0; i < size; i++) {
      dq.add(permutation[i]);
    }
  }

  private int getToken(Deque<Integer> dq,
                       List<Double> feature,
                       Random rnd) {
    if (dq.size() < windowSize) {
      generatePermutation(feature.size(), rnd, dq);
    }
    byte largest = 0;
    double largestValue = feature.get(dq.pollFirst());
    for (byte i = 1; i < windowSize; i++) {
      double value = feature.get(dq.pollFirst());
      if (value > largestValue) {
        largestValue = value;
        largest = i;
      }
    }
    return largest;
  }

  private int getWord(Deque<Integer> dq,
                      List<Double> feature,
                      Random rnd) {
    int result = 0;
    for (int i = 0; i < numTokensPerWord; i++) {
      result |= getToken(dq, feature, rnd) << 2 * i;
    }
    return result;
  }

  // Returns the "words" for a feature.
  // A word is compok
  private void getWordsForFeature(Set<String> output,
                                  String featureName,
                                  Map<String, List<Double>> denseFeatures) {
    List<Double> feature = denseFeatures.get(featureName);
    if (feature == null) {
      return;
    }
    assert (feature instanceof ArrayList);
    Random rnd = new Random(seed);
    Deque<Integer> dq = new ArrayDeque<>();
    for (int i = 0; i < numWordsPerFeature; i++) {
      String word = featureName + i + ':' + getWord(dq, feature, rnd);
      output.add(word);
    }
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, List<Double>> denseFeatures = featureVector.getDenseFeatures();
    if (denseFeatures == null) {
      return;
    }

    Set<String> output = new HashSet<>();

    for (String featureName : fieldNames) {
      getWordsForFeature(output, featureName, denseFeatures);
    }

    Util.optionallyCreateStringFeatures(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    stringFeatures.put(outputName, output);
  }
}
