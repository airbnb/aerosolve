package com.airbnb.aerosolve.core.perf;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 *
 */
public class FeatureRegistry {
  public static final int MAX_PREDICTED_FAMILY_SIZE = 100000;
  public static final int MIN_PREDICTED_FAMILIES = 10;
  private final Map<String, Family> families;
  private final AtomicInteger familyCount;

  public FeatureRegistry() {
    familyCount = new AtomicInteger(0);
    families = new ConcurrentHashMap<>(familyCapacity());
  }

  public int familyCapacity() {
    return Math.max(MIN_PREDICTED_FAMILIES, familyCount.get());
  }

  public Feature feature(String familyName, String featureName) {
    return family(familyName).feature(featureName);
  }

  public Family family(String familyName) {
    return families.computeIfAbsent(
        familyName,
        name -> new BasicFamily(name, familyCount.getAndIncrement()));
  }
}
