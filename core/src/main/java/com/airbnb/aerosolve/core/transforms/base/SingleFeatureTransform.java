package com.airbnb.aerosolve.core.transforms.base;

import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.typesafe.config.Config;
import lombok.Getter;

import javax.validation.constraints.NotNull;

/**
 *
 */
@SuppressWarnings("unchecked")
public abstract class SingleFeatureTransform<T extends SingleFeatureTransform>
    extends ConfigurableTransform<T>{

  @Getter
  @NotNull
  protected String inputFamilyName;

  @Getter
  @NotNull
  protected String inputFeatureName;

  @Getter
  protected String outputFamilyName;

  protected Family inputFamily;
  protected Feature inputFeature;
  protected Family outputFamily;

  protected SingleFeatureTransform() {
  }

  public T inputFamilyName(String name) {
    this.inputFamilyName = name;

    return (T) this;
  }

  public T outputFamilyName(String name) {
    this.outputFamilyName = name;
    return (T) this;
  }

  public T inputFeatureName(String name) {
    this.inputFeatureName = name;
    return (T) this;
  }

  @Override
  protected void setup() {
    super.setup();
    this.inputFamily = registry.family(inputFamilyName);
    this.inputFeature = inputFamily.feature(inputFeatureName);
    this.outputFamily = outputFamilyName == null
                        ? this.inputFamily
                        : registry.family(outputFamilyName);
  }

  @Override
  protected boolean checkPreconditions(MultiFamilyVector vector) {
    return super.checkPreconditions(vector) && vector.containsKey(inputFeature);
  }

  @Override
  public T configure(Config config, String key) {
    return (T)
        inputFamilyName(stringFromConfig(config, key, ".field1"))
        .inputFeatureName(stringFromConfig(config, key, inputFeatureKey()))
        .outputFamilyName(stringFromConfig(config, key, ".output", false));
  }

  protected String inputFeatureKey() {
    return ".value1";
  }

}
