package com.airbnb.aerosolve.core.features;

import lombok.Getter;

import java.util.*;

/*
  Features coming from differenct sources, and output as array[],
  FeatureMapping helps to save incoming feature in the right index of final output array[]
  it use incoming feature names array as key to locate the index.
  refer to ModelScorerTest.java as how to use FeatureMapping
 */
public class FeatureMapping {
  public final static int DEFAULT_SIZE = 100;
  @Getter
  private String[] names;
  @Getter
  private Integer[] types;
  private ArrayList<String> nameList;
  private ArrayList<Integer> typeList;
  @Getter
  private final Map<Object, Entry> mapping;

  public static final class Entry {
    int start;
    int length;
  }

  public FeatureMapping() {
    this(DEFAULT_SIZE);
  }

  public FeatureMapping(int size) {
    nameList = new ArrayList<>(size);
    typeList = new ArrayList<>(size);
    mapping = new HashMap<>(size);
  }

  // use name mapping array as key.
  public void add(String[] names, Integer type) {
    add(names, names, type);
  }

  public void add(Object c, String[] names, Integer type) {
    assert(names.length > 0);
    // should not add duplicated feature mapping
    assert(mapping.get(c) == null);
    Entry e = new Entry();
    e.start = nameList.size();
    e.length = names.length;
    Collections.addAll(nameList, names);
    for (int i = 0; i < names.length; i++) {
      typeList.add(type);
    }
    mapping.put(c, e);
  }

  public void add(Class c, List<ScoringFeature> features) {
    assert (features.size() > 0);
    Entry e = new Entry();
    e.start = nameList.size();
    e.length = features.size();
    for (ScoringFeature f : features) {
      nameList.add(f.getName());
      typeList.add(f.getType());
    }
    mapping.put(c, e);
  }

  public void finish() {
    names = new String[nameList.size()];
    nameList.toArray(names);
    types = new Integer[typeList.size()];
    typeList.toArray(types);
    nameList = null;
    typeList = null;
  }
}
