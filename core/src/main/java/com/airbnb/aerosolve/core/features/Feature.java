package com.airbnb.aerosolve.core.features;

import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import java.io.Serializable;

/**
 *
 */
public class Feature implements Comparable<Feature>, Serializable {
  private final Family family;
  private final String name;
  private final int hashCode;
  private final int index;

  // Package private because only Family can create features.
  Feature(Family family, String name, int index) {
    Preconditions.checkNotNull(family, "Family cannot be null for features");
    Preconditions.checkNotNull(name, "Name cannot be null for features");
    this.family = family;
    this.name = name;

    this.index = index;
    this.hashCode = 31 * family.hashCode() + name.hashCode();
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
    if (!(o instanceof Feature)) {
      return false;
    }
    Feature feature = (Feature) o;
    return Objects.equal(family, feature.family) &&
           feature.name().equals(this.name());
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
