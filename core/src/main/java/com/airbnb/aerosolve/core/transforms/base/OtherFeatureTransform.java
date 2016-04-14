package com.airbnb.aerosolve.core.transforms.base;

import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.typesafe.config.Config;
import lombok.Getter;

import javax.validation.constraints.NotNull;

/**
 *
 */
@SuppressWarnings("unchecked")
public abstract class OtherFeatureTransform<T extends OtherFeatureTransform>
    extends DualFamilyTransform<T> {

  @Getter
  @NotNull
  protected String otherFeatureName;

  protected Feature otherFeature;

  protected OtherFeatureTransform() {
  }

  public T otherFeatureName(String name) {
    this.otherFeatureName = name;
    return (T) this;
  }

  @Override
  protected void setup() {
    super.setup();
    otherFeature = otherFamily.feature(otherFeatureName);
  }

  @Override
  protected boolean checkPreconditions(MultiFamilyVector vector) {
    return super.checkPreconditions(vector) && vector.containsKey(otherFeature);
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
