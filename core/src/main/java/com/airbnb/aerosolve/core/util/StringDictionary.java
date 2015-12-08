package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.FeatureVector;

import lombok.Getter;
import lombok.Setter;

import java.io.Serializable;
import java.util.*;
import java.util.AbstractMap.SimpleEntry;
import java.util.Map.Entry;

/**
 * A class that maps strings to indices.
 */

public class StringDictionary implements Serializable {
  /**
   * The dictionary to maintain
   */
  
  protected Map<String, Map<String, Integer>> dictionary;
  @Getter
  protected int entryCount;

  public StringDictionary() {
    dictionary = new HashMap<>();
  }
  
  // Returns the index of the strings -1 if not present
  public int getIndex(String family, String feature) {
    Map<String, Integer> familyMap = dictionary.get(family);
    if (familyMap == null) {
      return -1;
    }
    Integer result = familyMap.get(feature);
    if (result == null) {
      return -1;
    }
    return result;
  }
   
  // Returns -1 if key exists, the index it was inserted if successful.
  public int possiblyAdd(String family, String feature) {
    Map<String, Integer> familyMap = dictionary.get(family);
    if (familyMap == null) {
      familyMap = new HashMap<>();
      dictionary.put(family, familyMap);
    }
    if (familyMap.containsKey(feature)) return -1;    
    int currIdx = entryCount;
    entryCount = entryCount + 1;
    familyMap.put(feature, currIdx);
    return currIdx;
  }
  
  public FloatVector makeVectorFromSparseFloats(Map<String, Map<String, Double>> sparseFloats) {
    FloatVector vec = new FloatVector(entryCount);
    for (Map.Entry<String, Map<String, Double>> kv : sparseFloats.entrySet()) {
      for (Map.Entry<String, Double> feat : kv.getValue().entrySet()) {
        int index = getIndex(kv.getKey(), feat.getKey());
        if (index >= 0) {
          vec.values[index] = feat.getValue().floatValue();
        }
      }
    }
    return vec;
  }
}