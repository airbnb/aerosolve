package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.perf.Family;
import com.airbnb.aerosolve.core.perf.Feature;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.types.DualFeatureTransform;
import com.typesafe.config.Config;
import java.util.List;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

/**
 * Quantizes the floatFeature named in "field1" with buckets in "bucket" before placing
 * it in the floatFeature named "output" subtracting the origin of the box.
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
public class MultiscaleGridContinuousTransform
    extends DualFeatureTransform<MultiscaleGridContinuousTransform> {

  private List<Double> buckets;

  @Override
  public MultiscaleGridContinuousTransform configure(Config config, String key) {
    return super.configure(config, key)
        .buckets(doubleListFromConfig(config, key, ".buckets", true));
  }

    @Override
  public void doTransform(MultiFamilyVector featureVector) {
    double v1 = featureVector.getDouble(inputFeature);
    double v2 = featureVector.getDouble(otherFeature);

    transformFeature(v1, v2, featureVector);
  }

  public void transformFeature(double v1,
                               double v2,
                               FeatureVector vector) {
    for (Double bucket : buckets) {
      transformFeature(v1, v2, bucket, outputFamily, vector);
    }
  }

  public static void transformFeature(double v1,
                                      double v2,
                                      double bucket,
                                      Family outputFamily,
                                      FeatureVector vector) {
    double mult1 = v1 / bucket;
    double q1 = bucket * Math.floor(mult1);
    double mult2 = v2 / bucket;
    double q2 = bucket * Math.floor(mult2);
    String bucketName = "[" + bucket + "]=(" + q1 + ',' + q2 + ')';
    vector.put(outputFamily.feature(bucketName + "@1"), v1 - q1);
    vector.put(outputFamily.feature(bucketName + "@2"), v2 - q2);
  }
}
