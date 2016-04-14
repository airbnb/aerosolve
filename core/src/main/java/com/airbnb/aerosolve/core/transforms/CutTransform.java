package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.Feature;
import com.airbnb.aerosolve.core.perf.FeatureValue;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.types.BoundedFeaturesTransform;

/**
 * remove features larger than upperBound or smaller than lowerBound
 */
@LegacyNames("cut_float")
public class CutTransform extends BoundedFeaturesTransform<CutTransform> {

  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    for (FeatureValue value : getInput(featureVector)) {
      Feature feature = value.feature();
      Feature outputFeature = produceOutputFeature(feature);
      if (featureVector.containsKey(feature)) {
        double v = featureVector.getDouble(feature);
        if (v > upperBound || v < lowerBound) {
          if (feature == outputFeature) {
            featureVector.remove(outputFeature);
          }
        } else if (feature != outputFeature) {
          featureVector.put(outputFeature, v);
        }
      }
    }
  }
}
