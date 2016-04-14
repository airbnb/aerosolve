package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.FamilyVector;
import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.ConfigurableTransform;
import com.typesafe.config.Config;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.Accessors;
import org.apache.commons.lang3.tuple.Pair;

/**
 *
 */
@LegacyNames({"delete_float_feature_family",
              "delete_string_feature_column",
              "delete_string_feature_family",
              "delete_float_feature",
              "delete_string_feature"})
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class DeleteTransform extends ConfigurableTransform<DeleteTransform> {
  private List<String> familyNames;
  private List<Map.Entry<String, String>> featureNames;
  private boolean deleteByPrefix = false;

  private List<Family> families;
  private List<Feature> features;

  @Override
  public DeleteTransform configure(Config config, String key) {
    DeleteTransform transform = familyNames(stringListFromConfig(config, key, ".fields", false));
    String familyName = stringFromConfig(config, key, ".field1", false);
    List<String> featureList = stringListFromConfig(config, key, ".keys", false);
    if (featureList != null && familyName != null) {
      List<Map.Entry<String, String>> featurePairs = new ArrayList<>();
      for (String featureName : featureList) {
        featurePairs.add(Pair.of(familyName, featureName));
      }
      transform.featureNames(featurePairs);
    }
    boolean deleteByPre = booleanFromConfig(config, key, ".delete_by_prefix");
    if (!deleteByPre) {
      String transformType = getTransformType(config, key);
      deleteByPre = transformType != null &&
                         (transformType.equals("delete_string_feature") ||
                          transformType.equals("delete_string_column"));
    }
    transform.deleteByPrefix(deleteByPre);
    return transform;
  }

  @Override
  protected void setup() {
    super.setup();
    if (familyNames != null) {
      families = new ArrayList<>();
      for (String familyName : familyNames) {
        families.add(registry.family(familyName));
      }
    }
    if (featureNames != null) {
      features = new ArrayList<>();
      for (Map.Entry<String, String> pair : featureNames) {
        features.add(registry.feature(pair.getKey(), pair.getValue()));
      }
    }
  }

  @Override
  protected void validate() {
    super.validate();
    if (familyNames == null && featureNames == null) {
      throw new IllegalArgumentException( "At least one of familyNames or featureNames must be set.");
    }
  }

  @Override
  protected void doTransform(MultiFamilyVector vector) {
    if (features != null) {
      for (Feature feature : features) {
        if (deleteByPrefix) {
          FamilyVector familyVector = vector.get(feature.family());
          if (familyVector == null) {
            continue;
          }
          List<Feature> toDelete = new ArrayList<>();
          for (FeatureValue value : familyVector) {
            if (value.feature().name().startsWith(feature.name())) {
              toDelete.add(value.feature());
            }
          }
          for (Feature deleteFeature : toDelete) {
            vector.removeDouble(deleteFeature);
          }
        } else {
          vector.removeDouble(feature);
        }
      }
    }

    if (families != null) {
      for (Family family : families) {
        vector.remove(family);
      }
    }
  }
}
