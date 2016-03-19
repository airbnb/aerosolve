package com.airbnb.aerosolve.core.features;

import lombok.Getter;

import java.util.HashSet;
import java.util.Set;

public class StringFamily extends FeatureFamily<String> {
  @Getter
  private final Set<String> features;

  public StringFamily(String familyName) {
    super(familyName);
    features = new HashSet<>();
  }

  @Override
  protected void put(String name, String feature) {
    features.add(feature);
  }

  public boolean add(String name, Object feature, Integer type) {
    if (type == Features.StringType) {
      return add(name, feature);
    } else {
      return add(name, (Boolean) feature);
    }
  }

  private boolean add(String name, Boolean feature) {
    String value = getBooleanFeatureAsString(name, feature);
    return add(name, value);
  }

  protected String getBooleanFeatureAsString(String name, Boolean feature) {
    if (feature) {
      return name + ":T";
    } else {
      return name + ":F";
    }
  }
}
