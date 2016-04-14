package com.airbnb.aerosolve.core.transforms.base;

import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.typesafe.config.Config;
import lombok.Getter;


/**
 *
 */
@SuppressWarnings("unchecked")
public abstract class DualFeatureTransform<T extends DualFeatureTransform>
    extends SingleFeatureTransform<T>{

  @Getter
  protected String otherFeatureName;

  @Getter
  protected String otherFamilyName;

  protected Family otherFamily;
  protected Feature otherFeature;

  protected DualFeatureTransform() {
  }

  public T otherFamilyName(String name) {
    this.otherFamilyName = name;
    return (T) this;
  }

  public T otherFeatureName(String name) {
    this.otherFeatureName = name;
    return (T) this;
  }

  @Override
  protected void setup() {
    super.setup();
    this.otherFamily = otherFamilyName == null ? inputFamily : registry.family(otherFamilyName);
    this.otherFeature = otherFamily.feature(otherFeatureName);
  }

  @Override
  protected boolean checkPreconditions(MultiFamilyVector vector) {
    return super.checkPreconditions(vector) && vector.containsKey(otherFeature);
  }

  public  T configure(Config config, String key) {
    return (T) super.configure(config, key)
        .otherFamilyName(stringFromConfig(config, key, ".field2", false))
        .otherFeatureName(stringFromConfig(config, key, otherFeatureKey()));
  }

  protected String otherFeatureKey() {
    return ".value2";
  }
}
