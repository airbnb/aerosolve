package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.DictionaryEntry;
import com.airbnb.aerosolve.core.DictionaryRecord;

import lombok.Getter;
import lombok.Setter;

import java.io.Serializable;
import java.util.*;
import java.util.AbstractMap.SimpleEntry;
import java.util.Map.Entry;

/**
 * A class that maps strings to indices. It can be used to map sparse
 * features (both sparse string and sparse float) into dense float vectors.
 */

public class StringDictionary implements Serializable {
  /**
   * The dictionary to maintain
   */
 
  @Getter
  protected DictionaryRecord dictionary;
 
  public StringDictionary() {
    dictionary = new DictionaryRecord();
    dictionary.setDictionary(new HashMap<String, Map<String, DictionaryEntry>>());
    dictionary.setEntryCount(0);
  }
  
  public StringDictionary(DictionaryRecord dict) {
    dictionary = dict;
  }
  
  // Returns the dictionary entry, null if not present
  public DictionaryEntry getEntry(String family, String feature) {
    Map<String, DictionaryEntry> familyMap = dictionary.dictionary.get(family);
    if (familyMap == null) {
      return null;
    }
    return familyMap.get(feature);
  }
   
  // Returns -1 if key exists, the index it was inserted if successful.
  public int possiblyAdd(String family, String feature, double mean, double scale) {
    Map<String, DictionaryEntry> familyMap = dictionary.dictionary.get(family);
    if (familyMap == null) {
      familyMap = new HashMap<>();
      dictionary.dictionary.put(family, familyMap);
    }
    if (familyMap.containsKey(feature)) return -1;
    DictionaryEntry entry = new DictionaryEntry();
    int currIdx = dictionary.getEntryCount();
    entry.setIndex(currIdx);
    entry.setMean(mean);
    entry.setScale(scale);
    dictionary.setEntryCount(currIdx + 1);
    familyMap.put(feature, entry);
    return currIdx;
  }
  
  public FloatVector makeVectorFromSparseFloats(Map<String, Map<String, Double>> sparseFloats) {
    FloatVector vec = new FloatVector(dictionary.getEntryCount());
    for (Map.Entry<String, Map<String, Double>> kv : sparseFloats.entrySet()) {
      for (Map.Entry<String, Double> feat : kv.getValue().entrySet()) {
        DictionaryEntry entry = getEntry(kv.getKey(), feat.getKey());
        if (entry != null) {
          vec.values[entry.index] = (float) entry.scale * (feat.getValue().floatValue() - (float) entry.mean);
        }
      }
    }
    return vec;
  }
}