package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.BaseFeaturesTransform;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;

/**
 * Computes the polynomial product of all values in field1
 * i.e. prod_i 1 + x_i
 * and places the result in outputName
 */
@NoArgsConstructor(access = AccessLevel.PACKAGE)
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
      prod *= 1.0 + value.value();
    }
    if (computedSomething) {
      featureVector.put(outputFeature, prod);
    }
  }
}
