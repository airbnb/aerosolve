package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.BoundedFeaturesTransform;
import java.util.ArrayList;
import java.util.List;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;

/**
 * remove features larger than upperBound or smaller than lowerBound
 */
@LegacyNames("cut_float")
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class CutTransform extends BoundedFeaturesTransform<CutTransform> {

  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    List<Feature> toDelete = new ArrayList<>();
    for (FeatureValue value : getInput(featureVector)) {
      double v = value.value();
      if (v > upperBound || v < lowerBound) {
        if (inputFamily.equals(outputFamily)) {
          toDelete.add(value.feature());
        }
      } else if (!inputFamily.equals(outputFamily)) {
        featureVector.put(produceOutputFeature(value.feature()), v);
      }
    }
    for (Feature feature : toDelete) {
      featureVector.remove(feature);
    }
  }
}
