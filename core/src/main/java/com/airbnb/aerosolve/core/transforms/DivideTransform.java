package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.Feature;
import com.airbnb.aerosolve.core.perf.FeatureValue;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.types.OtherFeatureTransform;
import com.typesafe.config.Config;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

/**
 * output = field1.keys / (field2.key2 + constant)
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
public class DivideTransform extends OtherFeatureTransform<DivideTransform> {
  private double constant = 0d;

  @Override
  public DivideTransform configure(Config config, String key) {
    return super.configure(config, key)
        .constant(doubleFromConfig(config, key, ".constant", 0.0d));
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
        double v = value.getDoubleValue();
        Feature outputFeature = produceOutputFeature(feature);
        featureVector.put(outputFeature, v * scale);
      }
    }
  }
}
