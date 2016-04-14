package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.Feature;
import com.airbnb.aerosolve.core.perf.FeatureValue;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.types.BoundedFeaturesTransform;

@LegacyNames("cap_float")
public class CapTransform extends BoundedFeaturesTransform<CapTransform> {

  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    for (FeatureValue value : getInput(featureVector)) {
      Feature feature = value.feature();
      if (featureVector.containsKey(feature)) {
        double v = value.getDoubleValue();
        featureVector.put(produceOutputFeature(feature),
                          Math.min(upperBound, Math.max(lowerBound, v)));
      }
    }
  }
}
