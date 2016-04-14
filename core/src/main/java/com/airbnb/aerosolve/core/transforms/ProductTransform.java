package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.Feature;
import com.airbnb.aerosolve.core.perf.FeatureValue;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.types.BaseFeaturesTransform;

/**
 * Computes the polynomial product of all values in field1
 * i.e. prod_i 1 + x_i
 * and places the result in outputName
 */
public class ProductTransform extends BaseFeaturesTransform<ProductTransform> {
  private Feature outputFeature;

  @Override
  protected void setup() {
    super.setup();
    outputFeature = outputFamily.feature("*");
  }

  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    Double prod = 1.0;
    boolean computedSomething = false;
    for (FeatureValue value : getInput(featureVector)) {
      computedSomething = true;
      prod *= 1.0 + value.getDoubleValue();
    }
    if (computedSomething) {
      featureVector.put(outputFeature, prod);
    }
  }
}
