package com.airbnb.aerosolve.core.transforms.types;

import com.airbnb.aerosolve.core.perf.Feature;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.typesafe.config.Config;
import lombok.Getter;

/**
 *
 */
@SuppressWarnings("unchecked")
public abstract class OtherFeatureTransform<T extends OtherFeatureTransform>
    extends DualFamilyTransform<T> {

  @Getter
  protected String otherFeatureName;

  protected Feature otherFeature;

  public T otherFeatureName(String name) {
    this.otherFeatureName = name;
    return (T) this;
  }

  @Override
  protected void setup() {
    super.setup();
    if (otherFeatureName != null) {
      otherFeature = otherFamily.feature(otherFeatureName);
    }
  }

  @Override
  protected boolean checkPreconditions(MultiFamilyVector vector) {
    boolean preconditions = super.checkPreconditions(vector);
    if (preconditions &&  otherFeature != null) {
      preconditions = vector.containsKey(otherFeature);
    }
    return preconditions;
  }

  @Override
  public T configure(Config config, String key) {
    return (T) super.configure(config, key)
        .otherFeatureName(stringFromConfig(config, key, otherFeatureKey(), false));
  }

  protected String otherFeatureKey() {
    return ".value2";
  }
}
