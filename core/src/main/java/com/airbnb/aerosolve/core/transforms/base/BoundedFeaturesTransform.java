package com.airbnb.aerosolve.core.transforms.base;

import com.typesafe.config.Config;

/**
 *
 */
@SuppressWarnings("unchecked")
public abstract class BoundedFeaturesTransform<T extends BoundedFeaturesTransform>
    extends BaseFeaturesTransform<T> {
  protected double lowerBound = -Double.MAX_VALUE;
  protected double upperBound = Double.MAX_VALUE;

  protected BoundedFeaturesTransform() {
  }

  public T lowerBound(double bound) {
    this.lowerBound = bound;
    return (T) this;
  }

  public T upperBound(double bound) {
    this.upperBound = bound;
    return (T) this;
  }

  @Override
  public T configure(Config config, String key) {
    return (T) super.configure(config, key)
        .lowerBound(doubleFromConfig(config, key, ".lower_bound", false, -Double.MAX_VALUE))
        .upperBound(doubleFromConfig(config, key, ".upper_bound", false, Double.MAX_VALUE));
  }

}
