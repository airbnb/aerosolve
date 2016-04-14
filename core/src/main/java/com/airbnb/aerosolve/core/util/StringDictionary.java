package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.DictionaryEntry;
import com.airbnb.aerosolve.core.DictionaryRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureValue;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import lombok.Getter;

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
    Map<String, DictionaryEntry> familyMap = dictionary.getDictionary().get(family);
    if (familyMap == null) {
      return null;
    }
    return familyMap.get(feature);
  }
   
  // Returns -1 if key exists, the index it was inserted if successful.
  public int possiblyAdd(Feature feature, double mean, double scale) {
    String familyName = feature.family().name();
    Map<String, DictionaryEntry> familyMap = dictionary.getDictionary().get(familyName);
    if (familyMap == null) {
      familyMap = new HashMap<>();
      dictionary.getDictionary().put(familyName, familyMap);
    }
    if (familyMap.containsKey(feature.name())) return -1;
    DictionaryEntry entry = new DictionaryEntry();
    int currIdx = dictionary.getEntryCount();
    entry.setIndex(currIdx);
    entry.setMean(mean);
    entry.setScale(scale);
    dictionary.setEntryCount(currIdx + 1);
    familyMap.put(feature.name(), entry);
    return currIdx;
  }
  
  public FloatVector makeVectorFromSparseFloats(FeatureVector vector) {
    FloatVector vec = new FloatVector(dictionary.getEntryCount());
    for (FeatureValue value : vector) {
      DictionaryEntry entry = getEntry(value.feature().family().name(), value.feature().name());
      if (entry != null) {
        vec.values[entry.getIndex()] = (float) (entry.getScale() *
                                                (value.value() - entry.getMean()));
      }
    }
    return vec;
  }
}