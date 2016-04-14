package com.airbnb.aerosolve.core.transforms.types;

import com.airbnb.aerosolve.core.perf.Family;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.typesafe.config.Config;

import javax.validation.constraints.NotNull;

/**
 *
 */
@SuppressWarnings("unchecked")
public abstract class DualFamilyTransform<T extends DualFamilyTransform>
    extends BaseFeaturesTransform<T> {
  @NotNull
  protected String otherFamilyName;

  protected Family otherFamily;

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
