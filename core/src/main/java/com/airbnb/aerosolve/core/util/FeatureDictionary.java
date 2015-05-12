package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.FeatureVector;
import lombok.Getter;
import lombok.Setter;

import java.io.Serializable;
import java.util.*;
import java.util.AbstractMap.SimpleEntry;
import java.util.Map.Entry;

/**
 * A class that maintains a dictionary of features and returns
 * the responses of a new feature to each element in the dictionary.
 */

public abstract class FeatureDictionary implements Serializable {
  /**
   * The dictionary to maintain
   */
  @Setter
  protected List<FeatureVector> dictionaryList;

  public FeatureDictionary() {
  }

  class EntryComparator implements Comparator<Entry<String, Double>> {
    public int compare(Entry<String, Double> e1,
                       Entry<String, Double> e2) {
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
   public abstract FeatureVector getKNearestNeighbors(
       KNearestNeighborsOptions options,
       FeatureVector featureVector);
}