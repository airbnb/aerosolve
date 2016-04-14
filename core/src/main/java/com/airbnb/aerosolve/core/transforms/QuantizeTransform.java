package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.BaseFeaturesTransform;
import com.airbnb.aerosolve.core.util.TransformUtil;
import com.google.common.collect.ImmutableSet;
import com.typesafe.config.Config;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;

/**
 * Created by hector_yee on 8/25/14.
 * TODO (Brad): Update doc
 * Multiplies the floatFeature named in "field1" with "scale" before placing
 * it in the stringFeature named "output"
 */
@LegacyNames({"custom_linear_log_quantize",
              "custom_multiscale_quantize",
              "linear_log_quantize",
              "multiscale_quantize"
})
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class QuantizeTransform extends BaseFeaturesTransform<QuantizeTransform> {
  /** Upper limit of each bucket to check if feature value falls in the bucket **/
  private static double[] LIMITS = {
      1.0, 10.0, 25.0, 50.0, 100.0, 400.0, 2000.0, 10000.0
  };

  /** Step size used for quantization, for the correponding limit **/
  private static double[] STEP_SIZES = {
      1.0 / 32.0, 0.125, 0.25, 5.0, 10.0, 25.0, 100.0, 250.0
  };

  /** Limit beyond which quantized value would be rounded to integer (ignoring decimals) **/
  private static double INTEGER_ROUNDING_LIMIT = 25.0;

  private Double scale;
  private QuantizeType type = QuantizeType.SIMPLE;
  private List<Double> buckets;
  private TreeMap<Double, Double> limitBucketPairs;

  @Setter(AccessLevel.NONE)
  private double upperLimit;

  @Override
  public QuantizeTransform configure(Config config, String key) {
    return super.configure(config, key)
        .type(figureOutType(config, key))
        .scale(doubleFromConfig(config, key, ".scale", false))
        .buckets(doubleListFromConfig(config, key, ".buckets", false))
        .limitBucketPairs(doubleTreeMapFromConfig(config, key, ".limit_bucket", false));
  }

  private QuantizeType figureOutType(Config config, String key) {
    String type = stringFromConfig(config, key, ".type", false);
    if (type != null) {
      return QuantizeType.valueOf(type.toUpperCase());
    }
    String transformType = getTransformType(config, key);
    // For legacy reasons
    if (transformType != null && transformType.equals("linear_log_quantize")) {
      return QuantizeType.LOG;
    }
    return QuantizeType.SIMPLE;
  }

  @Override
  protected void setup() {
    super.setup();
    if (limitBucketPairs != null) {
      upperLimit = limitBucketPairs.lastKey();
    }
  }

  // TODO (Brad): Validation

  @Override
  protected void doTransform(MultiFamilyVector featureVector) {

    for (FeatureValue value : getInput(featureVector)) {
      Set<Feature> resultFeatures;
      String featureName = value.feature().name();
      double featureValue = value.value();
      if (limitBucketPairs != null) {
        resultFeatures = customLogQuantize(featureName, featureValue);
      } else if (type == QuantizeType.LOG) {
        resultFeatures = logQuantize(featureName, featureValue);
      } else if (buckets != null) {
        resultFeatures = bucket(featureName, featureValue);
      } else if (scale != null) {
        resultFeatures = scale(featureName, featureValue);
      } else {
        // TODO (Brad): Log issue
        resultFeatures = ImmutableSet.of();
      }
      for (Feature feature : resultFeatures) {
        featureVector.putString(feature);
      }
    }
  }

  private Set<Feature> bucket(String featureName, double featureValue) {
    if (featureValue == 0.0) {
      return ImmutableSet.of(outputFamily.feature(featureName + "=0"));
    }

    Set<Feature> features = new HashSet<>();
    for (double bucket : buckets) {
      double quantized = TransformUtil.quantize(featureValue, bucket);
      features.add(outputFamily.feature(featureName + '[' + bucket + "]=" + quantized));
    }
    return features;
  }

  private Set<Feature> customLogQuantize(String featureName, double featureValue) {
    StringBuilder sb = new StringBuilder();
    sb.setLength(0);
    sb.append(featureName);
    boolean isValueNegative = false;
    if (featureValue < 0.0) {
      isValueNegative = true;
      featureValue = -featureValue;
    }

    if (featureValue < 1e-2) {
      sb.append("=0.0");
    } else {
      double limit;
      double bucket;
      if (featureValue >= upperLimit) {
        featureValue = upperLimit;
        bucket = limitBucketPairs.get(upperLimit);
      } else {
        limit = limitBucketPairs.higherKey(featureValue);
        bucket = limitBucketPairs.get(limit);
      }

      Double val = TransformUtil.quantize(featureValue, bucket) * 1000;

      sb.append('=');
      if (isValueNegative) {
        sb.append('-');
      }

      sb.append(val.intValue()/1000.0);
    }

    return ImmutableSet.of(outputFamily.feature(sb.toString()));
  }

  protected Set<Feature> scale(String featureName, double featureValue) {
    double dbl = featureValue * scale;
    return ImmutableSet.of(outputFamily.feature(featureName + '=' + (int) dbl));
  }

  private Set<Feature> logQuantize(String featureName, double featureValue) {
    StringBuilder sb = new StringBuilder();
    sb.append(featureName);
    sb.append('=');

    Double dbl = featureValue;
    if (dbl < 0.0) {
      sb.append('-');
      dbl = -dbl;
    }
    // At every stage we quantize roughly to a precision 10% of the magnitude.
    if (dbl < 1e-2) {
      sb.append('0');
    } else {
      boolean isQuantized = false;
      for (int i = 0; i < LIMITS.length; i++) {
        double limit = LIMITS[i];
        double stepSize = STEP_SIZES[i];
        if (limit > INTEGER_ROUNDING_LIMIT) {
          isQuantized = checkAndQuantize(sb, dbl, limit, stepSize, true);
        } else {
          isQuantized = checkAndQuantize(sb, dbl, limit, stepSize, false);
        }

        if (isQuantized) {
          break;
        }
      }

      if (! isQuantized) {
        Double exp = Math.log(dbl) / Math.log(2.0);
        Long val = 1L << exp.intValue();
        sb.append(val);
      }
    }

    return ImmutableSet.of(outputFamily.feature(sb.toString()));
  }

  private static boolean checkAndQuantize(StringBuilder sb,
                                          double featureValue,
                                          double limit,
                                          double stepSize,
                                          boolean integerRounding) {
    if (featureValue <= limit) {
      if (!integerRounding) {
        sb.append(TransformUtil.quantize(featureValue, stepSize));
      } else {
        sb.append(TransformUtil.quantize(featureValue, stepSize).intValue());
      }

      return true;
    }

    return false;
  }

  public enum QuantizeType {
    SIMPLE, LOG
  }
}
