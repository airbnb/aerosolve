package com.airbnb.aerosolve.core.features;

/**
 *
 */
public interface Family {

  int MIN_ALLOCATION = 4;
  int MAX_ALLOCATION = 64;

  int index();

  String name();

  void markDense();

  Feature feature(String featureName);

  int size();

  default int allocationSize() {
    return allocationSize(size());
  }

  static int allocationSize(int size) {
    int max = MIN_ALLOCATION;
    while (max < size && max < MAX_ALLOCATION)
    {
      max = max << 1;
    }
    return max;
  }

  Feature feature(int index);

  Feature cross(Feature left, Feature right, String separator);

  boolean isDense();
}
