package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.perf.Family;
import lombok.Value;
import lombok.experimental.Builder;

@Value
@Builder
public class KNearestNeighborsOptions {
  private final int numNearest;
  private final Family idKey;
  private final Family outputKey;
  private final Family featureKey;
}