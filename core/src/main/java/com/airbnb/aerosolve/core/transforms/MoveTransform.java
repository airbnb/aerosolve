package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.Feature;
import com.airbnb.aerosolve.core.perf.FeatureValue;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.types.BaseFeaturesTransform;
import com.airbnb.aerosolve.core.util.TransformUtil;
import com.typesafe.config.Config;
import java.util.List;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

/**
 * Moves named fields from one family to another. If keys are not specified, all keys are moved
 * from the float family.
 */
@LegacyNames({"move_float_to_string", "multiscale_move_float_to_string",
              "move_float_to_string_and_float"})
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
public class MoveTransform extends BaseFeaturesTransform<MoveTransform> {
  private Double bucket;
  private Double cap;
  private List<Double> buckets;
  private Double maxBucket;
  private Double minBucket;

  @Override
  public MoveTransform configure(Config config, String key) {
    return super.configure(config, key)
        .bucket(doubleFromConfig(config, key, ".bucket", null))
        .buckets(doubleListFromConfig(config, key, ".buckets", false))
        .maxBucket(doubleFromConfig(config, key, ".max_bucket", null))
        .minBucket(doubleFromConfig(config, key, ".min_bucket", null))
        .cap(doubleFromConfig(config, key, ".cap", 1e10));
  }

  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    for (FeatureValue value : getInput(featureVector)) {
      double dbl = value.getDoubleValue();
      if (dbl > cap) {
        dbl = cap;
      }

      boolean placed = false;
      if (bucket != null) {
        Double quantized = TransformUtil.quantize(dbl, bucket);
        if (minBucket == null ||  maxBucket == null
            || (quantized >= minBucket && quantized <= maxBucket)) {
          featureVector.putString(outputFamily.feature(value.feature().name() + '=' + quantized));
          placed = true;
        }
      } else if (buckets != null) {
        for (Double bucket : buckets) {
          Double quantized = TransformUtil.quantize(dbl, bucket);
          Feature outputFeature = outputFamily.feature(
              value.feature().name() + '[' + bucket + "]=" + quantized);
          featureVector.putString(outputFeature);
          placed = true;
        }
      }
      if (!placed) {
        // TODO (Brad): Question for Julian: Is it important that for move_float_to_string_and_float
        // the features end up in different families now that we can have Strings and Floats in
        // the same family?  Would it make sense to just leave it in the original family
        // if it's not inside the bounds?
        // We can handle it as is, but it's a little complex.
        featureVector.put(outputFamily.feature(value.feature().name()), value.getDoubleValue());
      }
      featureVector.remove(value.feature());
    }
  }
}
