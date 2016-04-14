package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.BaseFeaturesTransform;
import com.airbnb.aerosolve.core.util.TransformUtil;
import com.typesafe.config.Config;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.Accessors;

/**
 * Buckets float features and places them in a new float column.
 */
@LegacyNames("bucket_float")
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class BucketTransform extends BaseFeaturesTransform<BucketTransform> {
  private double bucket;

  public BucketTransform configure(Config config, String key) {
    return super.configure(config, key)
        .bucket(doubleFromConfig(config, key, ".bucket"));
  }

  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    for (FeatureValue value : getInput(featureVector)) {
      double dbl = TransformUtil.quantize(value.value(), bucket);
      double newVal = value.value() - dbl;
      String name = value.feature().name() + '[' + bucket + "]=" + dbl;
      featureVector.put(outputFamily.feature(name), newVal);
    }
  }
}