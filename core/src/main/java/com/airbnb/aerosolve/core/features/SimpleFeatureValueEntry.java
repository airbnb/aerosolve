package com.airbnb.aerosolve.core.features;

import java.io.Serializable;

/**
 *
 */
public class SimpleFeatureValueEntry extends SimpleFeatureValue
    implements FeatureValueEntry, Serializable {

  SimpleFeatureValueEntry(Feature feature, double value) {
    super(feature, value);
  }

  @Override
  @Deprecated
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


}
