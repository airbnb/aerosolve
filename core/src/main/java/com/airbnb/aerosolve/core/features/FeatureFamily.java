package com.airbnb.aerosolve.core.features;

import lombok.Getter;

public abstract class FeatureFamily <T> {
  @Getter
  private final String familyName;

  public FeatureFamily(String familyName) {
    this.familyName = familyName;
  }

  protected boolean isMyFamily(String name) {
    return true;
  }

  protected String nameTransform(String name) {
    return name;
  }

  public boolean add(String name, Object feature) {
    if (isMyFamily(name)) {
      put(nameTransform(name), (T) feature);
      return true;
    } else {
      return false;
    }
  }

  protected abstract void put(String name, T feature);
}
