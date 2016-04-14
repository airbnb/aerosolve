package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.DualFeatureTransform;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;

/**
 * id = fieldName1.key1
 * feature value = fieldName2.key2
 * output[ fieldname2 @ id ] = feature value
 * This transform is useful for making cross products of categorical features
 * e.g. leaf_id (say 123) and a continuous variable e.g. searches_at_leaf (say 4.0)
 * and making a new feature searches_at_leaf @ 123 = 4.0
 * The original searches_at_leaf feature can compare quantities at a global level
 * say searches in one market vs another market.
 * On the other hand searches_at_leaf @ 123 can tell you how the model changes
 * for searches at a particular place changing from day to day.
 */
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class StuffIdTransform extends DualFeatureTransform<StuffIdTransform> {

  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    double v1 = featureVector.getDouble(inputFeature);
    double v2 = featureVector.getDouble(otherFeature);

    String newname = otherFeature.name() + '@' + (long)v1;
    featureVector.put(outputFamily.feature(newname), v2);
  }

  @Override
  protected String inputFeatureKey() {
    return ".key1";
  }

  @Override
  protected String otherFeatureKey() {
    return ".key2";
  }
}
