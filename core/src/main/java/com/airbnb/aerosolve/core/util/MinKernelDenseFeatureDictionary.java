package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.FeatureVector;
import lombok.Setter;

import java.io.Serializable;
import java.util.*;
import java.util.AbstractMap.SimpleEntry;
import java.util.Map.Entry;

/**
 * A class that maintains a dictionary for dense features and returns
 * the responses of a new feature to each element in the dictionary.
 */

public class MinKernelDenseFeatureDictionary extends FeatureDictionary {
  /**
  /**
   * Calculates the Min Kernel distance to each dictionary element.
   * Returns the top K elements as a new sparse feature.
   */
  @Override
  public FeatureVector getKNearestNeighbors(
      KNearestNeighborsOptions options,
      FeatureVector featureVector) {
    FeatureVector result = new FeatureVector();
    Map<String, List<Double>> denseFeatures = featureVector.getDenseFeatures();

    if (denseFeatures == null) {
      return result;
    }
    PriorityQueue<SimpleEntry<String, Double>> pq = new PriorityQueue<>(
        options.getNumNearest() + 1,
        new EntryComparator());

    String idKey = options.getIdKey();

    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();
    String myId = featureVector.getStringFeatures()
        .get(idKey).iterator().next();

    for (FeatureVector supportVector : dictionaryList) {
      Double minKernel = FeatureVectorUtil.featureVectorMinKernel(featureVector,
                                                                  supportVector);
      Set<String> idSet = supportVector.getStringFeatures().get(idKey);
      String id = idSet.iterator().next();
      if (id == myId) continue;
      SimpleEntry<String, Double> entry = new SimpleEntry<String, Double>(id, minKernel);
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