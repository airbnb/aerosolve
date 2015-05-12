package com.airbnb.aerosolve.core.util;

import lombok.Getter;
import lombok.Setter;

public class KNearestNeighborsOptions {
  public KNearestNeighborsOptions() {
    numNearest = 5;
    idKey = "";
    outputKey = "";
    featureKey = "";
  }
  @Getter @Setter
  private int numNearest;
  @Getter @Setter
  public String idKey;
  @Getter @Setter
  public String outputKey;
  @Getter @Setter
  public String featureKey;
}