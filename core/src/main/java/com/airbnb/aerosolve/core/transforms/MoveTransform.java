package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.BaseFeaturesTransform;
import com.airbnb.aerosolve.core.util.TransformUtil;
import com.typesafe.config.Config;
import java.util.List;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.Accessors;

/**
 * Moves named fields from one family to another. If keys are not specified, all keys are moved
 * from the float family.
 */
@LegacyNames({"move_float_to_string",
              "multiscale_move_float_to_string",
              "move_float_to_string_and_float"})
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class MoveTransform extends BaseFeaturesTransform<MoveTransform> {
  private Double bucket;
  private Double cap;
  private List<Double> buckets;
  private Double maxBucket;
  private Double minBucket;

  @Override
  public MoveTransform configure(Config config, String key) {
    MoveTransform transform = super.configure(config, key);
    if (outputFamilyName == null) {
      // TODO (Brad): Make this ".output" in configs.
      transform.outputFamilyName(stringFromConfig(config, key, ".float_output"));
      // TODO (Brad): Think about ".string_output". Do we really need it?  It doesn't make sense
      // now that we don't distinguish strings and floats. But we may need multiple output families.
    }
    return transform.bucket(doubleFromConfig(config, key, ".bucket", false))
        .buckets(doubleListFromConfig(config, key, ".buckets", false))
        .maxBucket(doubleFromConfig(config, key, ".max_bucket", false))
        .minBucket(doubleFromConfig(config, key, ".min_bucket", false))
        .cap(doubleFromConfig(config, key, ".cap", false, 1e10));
  }

  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    for (FeatureValue value : getInput(featureVector)) {
      double dbl = value.value();
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
        // TODO (Brad): Question for Chris: Is it important that for move_float_to_string_and_float
        // the features end up in different families now that we can have Strings and Floats in
        // the same family?  Would it make sense to just leave it in the original family
        // if it's not inside the bounds?
        // We can handle it as is, but it feels a little complex and seems like it might be better
        // handled with two Moves.  One that keeps the features in the existing family and deletes
        // keys that match the interval.
        // Then we could have another Move that moves the family that remains if it's important it
        // have a different name.
        featureVector.put(outputFamily.feature(value.feature().name()), value.value());
      }
      featureVector.remove(value.feature());
    }
  }
}
