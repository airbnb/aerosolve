package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.FeatureVector;
import lombok.Setter;

import java.io.Serializable;
import java.util.*;
import java.util.AbstractMap.SimpleEntry;
import java.util.Map.Entry;

/**
 * A class that maintains a dictionary for sparse features and returns
 * the responses of a new feature to each element in the dictionary.
 */

public class LocalitySensitiveHashSparseFeatureDictionary extends FeatureDictionary {

  private Map<String, Set<Integer>> LSH;
  private boolean haveLSH;

  public LocalitySensitiveHashSparseFeatureDictionary() {
     haveLSH = false;
  }

  private int similarity(FeatureVector f1,
                            FeatureVector f2,
                            String featureKey) {
    Set<String> s1 = f1.getStringFeatures().get(featureKey);
    if (s1 == null) {
      return 0;
    }
    Set<String> s2 = f2.getStringFeatures().get(featureKey);
    if (s2 == null) {
      return 0;
    }
    Set<String> intersection = new HashSet<String>(s1);
    intersection.retainAll(s2);

    return intersection.size();
  }

  // Builds the hash table lookup for the LSH.
  private void buildHashTable(String featureKey) {
    LSH = new HashMap<>();
    assert(dictionaryList instanceof ArrayList);
    int size = dictionaryList.size();
    for (int i = 0; i < size; i++) {
      FeatureVector featureVector = dictionaryList.get(i);
      Set<String> keys = featureVector.getStringFeatures().get(featureKey);
      if (keys == null) {
        continue;
      }
      for (String key : keys) {
        Set<Integer> row = LSH.get(key);
        if (row == null) {
          row = new HashSet<>();
          LSH.put(key, row);
        }
        row.add(i);
      }
    }
  }

  // Returns all the candidates with a hash overlap.
  private Set<Integer> getCandidates(Set<String> keys) {
    Set<Integer> result = new HashSet<>();
    for (String key : keys) {
      Set<Integer> row = LSH.get(key);
      if (row != null) {
        result.addAll(row);
      }
    }
    return result;
  }

  @Override
  public FeatureVector getKNearestNeighbors(
      KNearestNeighborsOptions options,
      FeatureVector featureVector) {
    FeatureVector result = new FeatureVector();
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();

    if (stringFeatures == null) {
      return result;
    }

    String featureKey = options.getFeatureKey();
    Set<String> keys = stringFeatures.get(featureKey);
    if (keys == null) {
      return result;
    }

    if (!haveLSH) {
      buildHashTable(featureKey);
    }
    String idKey = options.getIdKey();
    PriorityQueue<SimpleEntry<String, Double>> pq = new PriorityQueue<>(
        options.getNumNearest() + 1,
        new EntryComparator());

    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();
    String myId = featureVector.getStringFeatures()
        .get(idKey).iterator().next();

    Set<Integer> candidates = getCandidates(keys);

    for (Integer candidate : candidates) {
      FeatureVector supportVector = dictionaryList.get(candidate);
      double sim = similarity(featureVector,
                                  supportVector,
                                  featureKey);
      Set<String> idSet = supportVector.getStringFeatures().get(idKey);
      String id = idSet.iterator().next();
      if (id == myId) {
        continue;
      }
      SimpleEntry<String, Double> entry = new SimpleEntry<String, Double>(id, sim);
      pq.add(entry);
      if (pq.size() > options.getNumNearest()) {
        pq.poll();
      }
    }

    HashMap<String, Double> newFeature = new HashMap<>();
    while (pq.peek() != null) {
      SimpleEntry<String, Double> entry = pq.poll();
      newFeature.put(entry.getKey(), entry.getValue());
    }
    floatFeatures.put(options.getOutputKey(), newFeature);
    result.setFloatFeatures(floatFeatures);

    return result;
  }
}