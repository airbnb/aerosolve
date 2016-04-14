package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.FeatureValue;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.types.OtherFeatureTransform;

/**
 * output = field1 - field2.key
 */
public class SubtractTransform extends OtherFeatureTransform<SubtractTransform> {

  @Override
  protected String produceOutputFeatureName(String featureName) {
    return featureName + '-' + otherFeature.name();
  }

  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    double sub = featureVector.getDouble(otherFeature);

    for (FeatureValue value : getInput(featureVector)) {
      double val = featureVector.getDouble(value.feature());
      featureVector.put(produceOutputFeature(value.feature()), val - sub);
    }
  }
}
