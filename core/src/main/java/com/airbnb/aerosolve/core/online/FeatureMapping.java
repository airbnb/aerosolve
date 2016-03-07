package com.airbnb.aerosolve.core.online;

import lombok.Getter;

import java.util.*;

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

  public void add(Object c, String[] names, Integer type) {
    assert (names.length > 0);
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
