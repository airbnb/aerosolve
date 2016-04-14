package com.airbnb.aerosolve.core.transforms.types;

import com.airbnb.aerosolve.core.perf.Family;
import com.airbnb.aerosolve.core.perf.Feature;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
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
        .inputFeatureName(stringFromConfig(config, key, ".value1"))
        .outputFamilyName(stringFromConfig(config, key, ".output", false));
  }
}
