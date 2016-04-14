package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.BoundedFeaturesTransform;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;

@LegacyNames("cap_float")
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class CapTransform extends BoundedFeaturesTransform<CapTransform> {

  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    for (FeatureValue value : getInput(featureVector)) {
      double v = value.value();
      featureVector.put(produceOutputFeature(value.feature()),
                        Math.min(upperBound, Math.max(lowerBound, v)));
    }
  }
}
