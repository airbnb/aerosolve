package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.transforms.types.BaseFeaturesTransform;

import com.airbnb.aerosolve.core.perf.FeatureValue;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import java.util.LinkedList;
import java.util.List;

/**
 * L2 normalizes a float feature
 */
@LegacyNames("normalize_float")
public class NormalizeTransform
    extends BaseFeaturesTransform<NormalizeTransform> {

  @Override
  protected void doTransform(MultiFamilyVector featureVector) {
    double norm = 0.0;
    List<FeatureValue> values = new LinkedList<>();
    for (FeatureValue value : getInput(featureVector)) {
      norm += value.getDoubleValue() * value.getDoubleValue();
      // TODO (Brad): May need to copy here due to re-using instances in the iterable.
      values.add(value);
    }
    if (norm > 0.0) {
      double scale = 1.0 / Math.sqrt(norm);
      // Not sure it's necessary but I'm storing the values to avoid mutating the vector
      // while iterating its values.
      for (FeatureValue value : values) {
        featureVector.put(value.feature(), value.getDoubleValue() * scale);
      }
    }
  }
}
