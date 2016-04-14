package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.OtherFeatureTransform;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;

/**
 * output = nearest of (field1, field2.key)
 */
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class NearestTransform extends OtherFeatureTransform<NearestTransform> {
  @Override
  protected void doTransform(MultiFamilyVector featureVector) {
    double sub = featureVector.getDouble(otherFeature);

    String nearest = "nothing";
    double bestDist = 1e10;

    for (FeatureValue value : getInput(featureVector)) {
      double dist = Math.abs(value.value() - sub);
      if (dist < bestDist) {
        nearest = value.feature().name();
        bestDist = dist;
      }
    }
    featureVector.putString(outputFamily.feature(otherFeature.name() + "~=" + nearest));
  }

  protected String otherFeatureKey() {
    return ".key";
  }
}
