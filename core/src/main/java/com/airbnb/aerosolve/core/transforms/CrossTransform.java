package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.models.AbstractModel;
import com.airbnb.aerosolve.core.perf.FamilyVector;
import com.airbnb.aerosolve.core.perf.Feature;
import com.airbnb.aerosolve.core.perf.FeatureValue;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.types.DualFamilyTransform;
import com.airbnb.aerosolve.core.util.TransformUtil;
import com.typesafe.config.Config;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

/**
 * Created by hector_yee on 8/25/14.
 * Takes the cross product of stringFeatures named in field1 and field2
 * and places it in a stringFeature with family name specified in output.
 */
@LegacyNames({"float_cross_float", "string_cross_float", "self_cross"})
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
public class CrossTransform extends DualFamilyTransform<CrossTransform>
    implements ModelAware<CrossTransform> {
  private Double bucket;
  private Double cap;
  private boolean putValue = true;
  private boolean ignoreNonModelFeatures = false;
  private AbstractModel model;

  public CrossTransform configure(Config config, String key) {
    return super.configure(config, key)
        .putValue(shouldPutValue(config, key))
        .bucket(doubleFromConfig(config, key, ".bucket", null))
        .cap(doubleFromConfig(config, key, ".cap", null))
        .ignoreNonModelFeatures(booleanFromConfig(config, key, ".ignoreNonModelFeatures"));
  }

  private boolean shouldPutValue(Config config, String key) {
    String type = getTransformType(config, key);
    return booleanFromConfig(config, key, ".putValue")
           || (type != null && type.endsWith("float"));
  }

  @Override
  protected void doTransform(MultiFamilyVector featureVector) {
    FamilyVector otherFamilyVector = featureVector.get(otherFamily);
    for (FeatureValue value : getInput(featureVector)) {
      double doubleVal = value.getDoubleValue();
      if (cap != null) {
        doubleVal = Math.min(cap, doubleVal);
      }
      double quantizedVal = 0d;
      if (bucket != null) {
        quantizedVal = TransformUtil.quantize(doubleVal, bucket);
      }
      for (FeatureValue otherValue : otherFamilyVector) {
        if (value.feature() == otherValue.feature()) {
          continue;
        }
        String separator = "^";
        if (bucket != null) {
          separator = "=" + quantizedVal + "^";
        }
        Feature outputFeature = outputFamily.cross(value.feature(), otherValue.feature(), separator);
        if (!ignoreNonModelFeatures || model == null || model.needsFeature(outputFeature)) {
          if (putValue) {
            featureVector.put(outputFeature, doubleVal);
          } else {
            featureVector.putString(outputFeature);
          }
        }
      }
    }
  }
}
