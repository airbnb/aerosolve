package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.FeatureValue;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.types.OtherFeatureTransform;

/**
 * output = nearest of (field1, field2.key)
 */
public class NearestTransform extends OtherFeatureTransform<NearestTransform> {
  @Override
  protected void doTransform(MultiFamilyVector featureVector) {
    double sub = featureVector.getDouble(otherFeature);

    String nearest = "nothing";
    double bestDist = 1e10;

    for (FeatureValue value : getInput(featureVector)) {
      double dist = Math.abs(value.getDoubleValue() - sub);
      if (dist < bestDist) {
        nearest = value.feature().name();
        bestDist = dist;
      }
    }
    featureVector.putString(outputFamily.feature(otherFeature.name() + "~=" + nearest));
  }
}
