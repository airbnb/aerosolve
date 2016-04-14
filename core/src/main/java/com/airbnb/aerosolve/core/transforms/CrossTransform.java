package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.models.AbstractModel;
import com.airbnb.aerosolve.core.features.FamilyVector;
import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.DualFamilyTransform;
import com.airbnb.aerosolve.core.util.TransformUtil;
import com.typesafe.config.Config;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.Setter;
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
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class CrossTransform extends DualFamilyTransform<CrossTransform>
    implements ModelAware<CrossTransform> {
  private Double bucket;
  private Double cap;
  private boolean putValue = true;
  private boolean ignoreNonModelFeatures = false;

  private AbstractModel model;

  @Setter(AccessLevel.NONE)
  private boolean selfCross = false;

  public CrossTransform configure(Config config, String key) {
    return super.configure(config, key)
        .bucket(doubleFromConfig(config, key, ".bucket", false))
        .cap(doubleFromConfig(config, key, ".cap", false))
        .putValue(shouldPutValue(config, key))
        .ignoreNonModelFeatures(booleanFromConfig(config, key, ".ignore_non_model_features"));
  }

  private boolean shouldPutValue(Config config, String key) {
    String type = getTransformType(config, key);
    return booleanFromConfig(config, key, ".putValue")
           || (type != null && type.endsWith("float"))
           || bucket != null || cap != null;
  }

  @Override
  protected void setup() {
    super.setup();
    selfCross = otherFamily == inputFamily;
  }

  @Override
  protected void doTransform(MultiFamilyVector featureVector) {
    FamilyVector otherFamilyVector = featureVector.get(otherFamily);
    for (FeatureValue value : getInput(featureVector)) {
      String separator = "^";
      if (putValue) {
        double doubleVal = value.value();
        if (cap != null) {
          doubleVal = Math.min(cap, doubleVal);
        }
        if (bucket != null) {
          double quantizedVal = TransformUtil.quantize(doubleVal, bucket);
          separator = "=" + quantizedVal + "^";
        }
      }
      for (FeatureValue otherValue : otherFamilyVector) {
        if (value.feature().equals(otherValue.feature())) {
          continue;
        }
        Feature outputFeature = outputFamily.cross(value.feature(), otherValue.feature(), separator);
        boolean neededForModel = !ignoreNonModelFeatures
                                    || model == null
                                    || model.needsFeature(outputFeature);
        // For self crosses, there is no reason to cross both ways.
        boolean neededIfSelfCross = !selfCross
                                     || value.feature().compareTo(otherValue.feature()) < 0;
        if (neededForModel && neededIfSelfCross) {
          if (putValue) {
            featureVector.put(outputFeature, otherValue.value());
          } else {
            featureVector.putString(outputFeature);
          }
        }
      }
    }
  }
}
