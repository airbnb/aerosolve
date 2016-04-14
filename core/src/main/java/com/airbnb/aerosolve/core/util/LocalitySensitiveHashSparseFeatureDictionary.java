package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.perf.Family;
import com.airbnb.aerosolve.core.perf.FamilyVector;
import com.airbnb.aerosolve.core.perf.FastMultiFamilyVector;
import com.airbnb.aerosolve.core.perf.Feature;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.perf.FeatureValue;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.Sets;
import java.util.AbstractMap.SimpleEntry;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * A class that maintains a dictionary for sparse features and returns
 * the responses of a new feature to each element in the dictionary.
 */

public class LocalitySensitiveHashSparseFeatureDictionary extends FeatureDictionary {

  private Map<Feature, Set<Integer>> LSH;

  public LocalitySensitiveHashSparseFeatureDictionary(FeatureRegistry registry) {
    super(registry);
  }

  private int similarity(MultiFamilyVector f1,
                         MultiFamilyVector f2,
                          Family featureKey) {
    FamilyVector fam1 = f1.get(featureKey);
    FamilyVector fam2 = f2.get(featureKey);
    if (fam1 == null || fam2 == null) {
      return 0;
    }
    return Sets.intersection(fam1.keySet(), fam2.keySet()).size();
  }

  // Builds the hash table lookup for the LSH.
  private void buildHashTable(Family featureKey) {
    LSH = new HashMap<>();
    assert(dictionaryList instanceof ArrayList);
    int size = dictionaryList.size();
    for (int i = 0; i < size; i++) {
      MultiFamilyVector featureVector = dictionaryList.get(i);
      FamilyVector vec = featureVector.get(featureKey);
      if (vec == null) {
        continue;
      }
      for (FeatureValue value : vec) {
        Set<Integer> row = LSH.computeIfAbsent(value.feature(),
                                               f -> new HashSet<>());
        row.add(i);
      }
    }
  }

  // Returns all the candidates with a hash overlap.
  private Set<Integer> getCandidates(FamilyVector vector) {
    Set<Integer> result = new HashSet<>();
    for (FeatureValue value : vector) {
      Set<Integer> row = LSH.get(value.feature());
      if (row != null) {
        result.addAll(row);
      }
    }
    return result;
  }

  @Override
  public MultiFamilyVector getKNearestNeighbors(
      KNearestNeighborsOptions options,
      FeatureVector featureVector) {
    MultiFamilyVector result = new FastMultiFamilyVector(registry);

    if (!(featureVector instanceof MultiFamilyVector)) {
      return result;
    }

    MultiFamilyVector vector = (MultiFamilyVector) featureVector;

    Family featureKey = options.getFeatureKey();

    FamilyVector keys = vector.get(featureKey);
    if (keys == null) {
      return result;
    }

    if (LSH == null) {
      buildHashTable(featureKey);
    }
    Family idKey = options.getIdKey();
    PriorityQueue<SimpleEntry<Feature, Double>> pq = new PriorityQueue<>(
        options.getNumNearest() + 1,
        new EntryComparator());

    FamilyVector idVector = vector.get(idKey);
    if (idVector == null || idVector.isEmpty()) {
      return result;
    }

    Feature myId = idVector.iterator().next().feature();

    Set<Integer> candidates = getCandidates(keys);

    for (Integer candidate : candidates) {
      MultiFamilyVector supportVector = dictionaryList.get(candidate);
      double sim = similarity(vector,
                              supportVector,
                              featureKey);

      FamilyVector idSet = supportVector.get(idKey);
      if (idSet == null) {
        continue;
      }
      Feature id = idSet.iterator().next().feature();
      if (id == myId) {
        continue;
      }
      pq.add(new SimpleEntry<>(id, sim));
      if (pq.size() > options.getNumNearest()) {
        pq.poll();
      }
    }

    Family outputFamily = options.getOutputKey();
    while (pq.peek() != null) {
      SimpleEntry<Feature, Double> entry = pq.poll();
      Feature outputFeature = outputFamily.feature(entry.getKey().name());
      result.put(outputFeature, (double) entry.getValue());
    }

    return result;
  }
}