package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.ConfigurableTransform;
import com.typesafe.config.Config;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;

/**
 * Given a fieldName1, low, upper key
 * Remaps fieldName2's key2 value such that low = 0, upper = 1.0 thus approximating
 * the percentile using linear interpolation.
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class ApproximatePercentileTransform
    extends ConfigurableTransform<ApproximatePercentileTransform> {

  protected String lowFamilyName;
  protected String lowFeatureName;
  protected String upperFamilyName;
  protected String upperFeatureName;
  protected String valueFamilyName;
  protected String valueFeatureName;
  protected String outputFamilyName;
  protected String outputFeatureName;
  protected double minDiff;

  @Setter(AccessLevel.NONE)
  protected Feature upperFeature;
  @Setter(AccessLevel.NONE)
  protected Feature lowFeature;
  @Setter(AccessLevel.NONE)
  protected Feature valueFeature;
  @Setter(AccessLevel.NONE)
  protected Feature outputFeature;

  @Override
  public ApproximatePercentileTransform configure(Config config, String key) {
    return lowFamilyName(stringFromConfig(config, key, ".field1"))
        .lowFeatureName(stringFromConfig(config, key, ".low"))
        .upperFamilyName(stringFromConfig(config, key, ".field1"))
        .upperFeatureName(stringFromConfig(config, key, ".upper"))
        .valueFamilyName(stringFromConfig(config, key, ".field2"))
        .valueFeatureName(stringFromConfig(config, key, ".key2"))
        .minDiff(doubleFromConfig(config, key, ".minDiff"))
        .outputFamilyName(stringFromConfig(config, key, ".output"))
        .outputFeatureName(stringFromConfig(config, key, ".outputKey"));
  }

  protected void setup() {
    super.setup();
    this.lowFeature = registry.feature(lowFamilyName, lowFeatureName);
    this.upperFeature = registry.feature(upperFamilyName, upperFeatureName);
    this.valueFeature = registry.feature(valueFamilyName, valueFeatureName);
    this.outputFeature = registry.feature(outputFamilyName, outputFeatureName);
  }

  @Override
  protected boolean checkPreconditions(MultiFamilyVector vector) {
    return super.checkPreconditions(vector) &&
           vector.containsKey(lowFeature) &&
           vector.containsKey(upperFeature) &&
           vector.containsKey(valueFeature);
  }

  @Override
  protected void doTransform(MultiFamilyVector vector) {

    double low = vector.getDouble(lowFeature);
    double upper = vector.getDouble(upperFeature);
    double val = vector.getDouble(valueFeature);

    // Abstain if the percentiles are too close.
    double denom = upper - low;
    if (denom < minDiff) {
      return;
    }

    double outVal;
    if (val <= low) {
      outVal = 0.0d;
    } else if (val >= upper) {
      outVal = 1.0d;
    } else {
      outVal = (val - low) / denom;
    }

    vector.put(outputFeature, outVal);
  }
}
