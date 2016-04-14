package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import java.io.Serializable;
import java.util.Comparator;
import java.util.List;
import java.util.Map.Entry;
import lombok.Setter;

/**
 * A class that maintains a dictionary of features and returns
 * the responses of a new feature to each element in the dictionary.
 */

public abstract class FeatureDictionary implements Serializable {

  protected final FeatureRegistry registry;
  /**
   * The dictionary to maintain
   */
  @Setter
  protected List<MultiFamilyVector> dictionaryList;

  public FeatureDictionary(FeatureRegistry registry) {
    this.registry = registry;
  }

  class EntryComparator implements Comparator<Entry<?, Double>> {
    public int compare(Entry<?, Double> e1,
                       Entry<?, Double> e2) {
      if (e1.getValue() > e2.getValue()) {
        return 1;
      } else if (e1.getValue() < e2.getValue()) {
        return -1;
      }
      return 0;
    }
  }

  /**
   * Returns the k-nearest neighbors as floatFeatures in featureVector.
   */
   public abstract MultiFamilyVector getKNearestNeighbors(
       KNearestNeighborsOptions options,
       FeatureVector featureVector);
}