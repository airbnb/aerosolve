package com.airbnb.aerosolve.core.perf;

import lombok.ToString;

/**
 *
 */
@ToString
public class SimpleFeatureValueEntry implements FeatureValueEntry {
  private Feature feature;
  private double value;

  public SimpleFeatureValueEntry(Feature feature, double value) {
    this.feature = feature;
    this.value = value;
  }

  public SimpleFeatureValueEntry() {
  }

  @Override
  public Double getValue() {
    return value;
  }

  @Override
  public double setValue(double value) {
    this.value = value;
    return this.value;
  }

  @Override
  public double getDoubleValue() {
    return value;
  }

  @Override
  public Feature getKey() {
    return feature;
  }

  @Override
  public Double setValue(Double value) {
    return setValue((double)value);
  }

  @Override
  public Feature feature() {
    return feature;
  }

  @Override
  public Feature setFeature(Feature feature) {
    this.feature = feature;
    return this.feature;
  }

  public static SimpleFeatureValueEntry of(Feature feature, double value) {
    return new SimpleFeatureValueEntry(feature, value);
  }
}
