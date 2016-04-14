package com.airbnb.aerosolve.core.features;

import java.util.Map;
import java.util.Set;
import lombok.Builder;
import lombok.Value;

/**
 *
 */
public interface NamingConvention {

  NamingConventionResult features(String name, Object value, FeatureRegistry registry);

  @Value
  @Builder
  class NamingConventionResult {
    private final Set<Feature> stringFeatures;
    private final Set<FeatureValue> doubleFeatures;
    private final Map<Family, double[]> denseFeatures;
  }
}
