package com.airbnb.aerosolve.core.transforms.base;

import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.typesafe.config.Config;

/**
 *
 */
@SuppressWarnings("unchecked")
public abstract class DualFamilyTransform<T extends DualFamilyTransform>
    extends BaseFeaturesTransform<T> {
  protected String otherFamilyName;

  protected Family otherFamily;

  protected DualFamilyTransform() {
  }

  public T otherFamilyName(String name) {
    this.otherFamilyName = name;
    return (T) this;
  }

  @Override
  protected void setup() {
    super.setup();
    this.otherFamily = otherFamilyName == null ? inputFamily : registry.family(otherFamilyName);
  }

  @Override
  protected boolean checkPreconditions(MultiFamilyVector vector) {
    return super.checkPreconditions(vector) && vector.contains(otherFamily);
  }

  public  T configure(Config config, String key) {
    return (T) super.configure(config, key)
         .otherFamilyName(stringFromConfig(config, key, ".field2", false));
  }
}
