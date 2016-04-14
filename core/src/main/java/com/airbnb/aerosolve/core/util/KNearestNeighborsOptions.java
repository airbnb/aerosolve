package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.features.Family;
import lombok.Builder;
import lombok.Value;

@Value
@Builder
public class KNearestNeighborsOptions {
  private final int numNearest;
  private final Family idKey;
  private final Family outputKey;
  private final Family featureKey;
}