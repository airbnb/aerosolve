package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.DualFeatureTransform;
import com.airbnb.aerosolve.core.util.TransformUtil;
import com.typesafe.config.Config;
import java.util.List;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.Accessors;

/**
 * Created by hector_yee on 8/25/14.
 * Quantizes the floatFeature named in "field1" with buckets in "bucket" before placing
 * it in the stringFeature named "output"
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class MultiscaleGridQuantizeTransform
    extends DualFeatureTransform<MultiscaleGridQuantizeTransform> {
  private List<Double> buckets;

  @Override
  public MultiscaleGridQuantizeTransform configure(Config config, String key) {
    return super.configure(config, key)
        .buckets(doubleListFromConfig(config, key, ".buckets", true));
  }

  @Override
  public void doTransform(MultiFamilyVector featureVector) {

    double v1 = featureVector.getDouble(inputFeature);
    double v2 = featureVector.getDouble(otherFeature);

    transformFeature(v1, v2, buckets, outputFamily, featureVector);
  }

  public static void transformFeature(double v1, double v2, List<Double> buckets,
                                      Family outputFamily,
                                      FeatureVector vector) {
    for (Double bucket : buckets) {
      transformFeature(v1, v2, bucket, outputFamily, vector);
    }
  }

  public static void transformFeature(double v1, double v2, double bucket,
                                      Family outputFamily, FeatureVector vector) {
    double q1 = TransformUtil.quantize(v1, bucket);
    double q2 = TransformUtil.quantize(v2, bucket);
    vector.putString(outputFamily.feature("[" + bucket + "]=(" + q1 + ',' + q2 + ')'));
  }
}
