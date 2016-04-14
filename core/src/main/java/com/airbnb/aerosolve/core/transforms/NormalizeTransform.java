package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.transforms.base.BaseFeaturesTransform;

import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import java.util.LinkedList;
import java.util.List;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;

/**
 * L2 normalizes a float feature
 */
@LegacyNames("normalize_float")
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class NormalizeTransform
    extends BaseFeaturesTransform<NormalizeTransform> {

  @Override
  protected void doTransform(MultiFamilyVector featureVector) {
    double norm = 0.0;
    List<FeatureValue> values = new LinkedList<>();
    for (FeatureValue value : getInput(featureVector)) {
      norm += value.value() * value.value();
      // TODO (Brad): May need to copy here due to re-using instances in the iterable.
      values.add(value);
    }
    if (norm > 0.0) {
      double scale = 1.0 / Math.sqrt(norm);
      // Not sure it's necessary but I'm storing the values to avoid mutating the vector
      // while iterating its values.
      for (FeatureValue value : values) {
        featureVector.put(value.feature(), value.value() * scale);
      }
    }
  }
}
