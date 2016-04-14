package com.airbnb.aerosolve.core.features;

import java.io.Serializable;
import lombok.Data;
import lombok.experimental.Accessors;

/**
 *
 */
@Data
@Accessors(fluent = true, chain = false)
public class SimpleFeatureValue implements FeatureValue, Serializable {
  protected Feature feature;
  protected double value;

  SimpleFeatureValue(Feature feature, double value) {
    this.feature = feature;
    this.value = value;
  }

  public static SimpleFeatureValue of(Feature feature, double value) {
    return new SimpleFeatureValue(feature, value);
  }
}
