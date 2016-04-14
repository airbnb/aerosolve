package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.OtherFeatureTransform;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;

/**
 * output = field1 - field2.key
 */
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class SubtractTransform extends OtherFeatureTransform<SubtractTransform> {

  @Override
  protected String produceOutputFeatureName(String featureName) {
    return featureName + '-' + otherFeatureName;
  }

  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    double sub = featureVector.getDouble(otherFeature);

    for (FeatureValue value : getInput(featureVector)) {
      double val = featureVector.getDouble(value.feature());
      featureVector.put(produceOutputFeature(value.feature()), val - sub);
    }
  }

  @Override
  protected String otherFeatureKey() {
    return ".key2";
  }
}
