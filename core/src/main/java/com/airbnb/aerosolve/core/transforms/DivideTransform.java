package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.OtherFeatureTransform;
import com.typesafe.config.Config;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.Accessors;

/**
 * output = field1.keys / (field2.key2 + constant)
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class DivideTransform extends OtherFeatureTransform<DivideTransform> {
  private double constant = 0d;

  @Override
  public DivideTransform configure(Config config, String key) {
    return super.configure(config, key)
        .constant(doubleFromConfig(config, key, ".constant", false, 0.0d));
  }

  @Override
  protected String otherFeatureKey() {
    return ".key2";
  }

  @Override
  public String produceOutputFeatureName(String name) {
    return name + "-d-" + otherFeatureName;
  }

  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    double div = featureVector.getDouble(otherFeature);

    double scale = 1.0 / (constant + div);

    for (FeatureValue value : getInput(featureVector)) {
      Feature feature = value.feature();
      if (featureVector.containsKey(feature)) {
        double v = value.value();
        Feature outputFeature = produceOutputFeature(feature);
        featureVector.put(outputFeature, v * scale);
      }
    }
  }
}
