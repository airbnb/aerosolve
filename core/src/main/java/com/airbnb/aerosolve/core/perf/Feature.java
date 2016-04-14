package com.airbnb.aerosolve.core.perf;

import com.google.common.base.Objects;
import com.google.common.base.Preconditions;

/**
 *
 */
public class Feature implements Comparable<Feature> {
  private final Family family;
  private final String name;
  private final int hashCode;
  private final int index;

  // Package private because only Family can create features.  This gives us referential
  // comparison for fast equals.
  Feature(Family family, String name, int index) {
    Preconditions.checkNotNull(family, "Family cannot be null for features");
    Preconditions.checkNotNull(name, "Name cannot be null for features");
    this.family = family;
    this.name = name;

    this.index = index;
    this.hashCode = FeatureRegistry.MAX_PREDICTED_FAMILY_SIZE * family.hashCode() + index;
  }

  public Family family() {
    return family;
  }

  public String name() {
    return name;
  }

  public int index() {
    return index;
  }

  @Override
  public int hashCode() {
    return hashCode;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    Feature feature = (Feature) o;
    return Objects.equal(family, feature.family) &&
           feature.index == this.index;
  }

  @Override
  public int compareTo(Feature other) {
    int famCompare = this.family().index() - other.family().index();
    if (famCompare != 0) {
      return famCompare;
    }
    return this.index() - other.index();
  }

  @Override
  public String toString() {
    return String.format("%s :: %s", family, name);
  }
}
