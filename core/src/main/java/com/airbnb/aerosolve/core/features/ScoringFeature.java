package com.airbnb.aerosolve.core.features;

import lombok.Getter;

public class ScoringFeature implements Comparable<ScoringFeature> {
  @Getter
  private String name;
  @Getter
  private Integer type;

  public ScoringFeature(String name, Integer type) {
    this.name = name;
    this.type = type;
  }
  public int compareTo(ScoringFeature other) {
    return this.getName().compareTo(other.getName());
  }
}
