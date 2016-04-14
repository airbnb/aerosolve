package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.perf.Family;
import com.airbnb.aerosolve.core.perf.FamilyVector;
import com.airbnb.aerosolve.core.perf.FastMultiFamilyVector;
import com.airbnb.aerosolve.core.perf.Feature;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.perf.FeatureValue;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import java.util.AbstractMap.SimpleEntry;
import java.util.PriorityQueue;

/**
 * A class that maintains a dictionary for dense features and returns
 * the responses of a new feature to each element in the dictionary.
 */

public class MinKernelDenseFeatureDictionary extends FeatureDictionary {

  public MinKernelDenseFeatureDictionary(FeatureRegistry registry) {
    super(registry);
  }

  /**
   * Calculates the Min Kernel distance to each dictionary element.
   * Returns the top K elements as a new sparse feature.
   */
  @Override
  public MultiFamilyVector getKNearestNeighbors(
      KNearestNeighborsOptions options,
      FeatureVector featureVector) {
    MultiFamilyVector result = new FastMultiFamilyVector(registry);

    if (!(featureVector instanceof MultiFamilyVector)) {
      return result;
    }

    MultiFamilyVector vector = (MultiFamilyVector) featureVector;

    PriorityQueue<SimpleEntry<Feature, Double>> pq = new PriorityQueue<>(
        options.getNumNearest() + 1,
        new EntryComparator());

    Family idKey = options.getIdKey();

    FamilyVector idVector = vector.get(idKey);
    if (idVector == null || idVector.isEmpty()) {
      return result;
    }

    Feature myId = idVector.iterator().next().feature();

    for (MultiFamilyVector supportVector : dictionaryList) {
      double minKernel = FeatureVectorUtil.featureVectorMinKernel(featureVector,
                                                                  supportVector);
      FamilyVector idSet = supportVector.get(idKey);
      if (idSet == null) {
        continue;
      }
      Feature id = idSet.iterator().next().feature();
      if (id == myId) {
        continue;
      }
      SimpleEntry<Feature, Double> entry = new SimpleEntry<>(id, minKernel);
      pq.add(entry);
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