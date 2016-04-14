package com.airbnb.aerosolve.core.perf;

/**
 *
 */
public interface Family {

  int MIN_ALLOCATION = 4;
  int MAX_ALLOCATION = 64;

  int index();

  String name();

  void makeDense();

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

  default boolean isEqual(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    Family family = (Family) o;
    return family.index() == index();
  }

  default int getHashCode() {
    return index();
  }

  Feature feature(int index);

  Feature cross(Feature left, Feature right, String separator);
}
