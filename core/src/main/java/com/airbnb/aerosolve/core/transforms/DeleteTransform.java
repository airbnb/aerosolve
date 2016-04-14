package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.Family;
import com.airbnb.aerosolve.core.perf.Feature;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.types.ConfigurableTransform;
import com.google.common.collect.ImmutableSet;
import com.typesafe.config.Config;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;

import javax.validation.ConstraintViolationException;

/**
 *
 */
@LegacyNames({"delete_float_feature_family",
              "delete_string_feature_column",
              "delete_string_feature_family",
              "delete_float_feature",
              "delete_string_feature"})
@Accessors(fluent = true, chain = true)
public class DeleteTransform extends ConfigurableTransform<DeleteTransform> {
  @Getter
  @Setter
  private List<String> familyNames;
  @Getter
  @Setter
  private Map<String, String> featureNames;

  private List<Family> families;
  private List<Feature> features;

  @Override
  public DeleteTransform configure(Config config, String key) {
    DeleteTransform transform = familyNames(stringListFromConfig(config, key, ".fields", false));
    String familyName = stringFromConfig(config, key, ".field1", false);
    List<String> featureList = stringListFromConfig(config, key, ".keys", false);
    if (featureList != null && familyName != null) {
      Map<String, String> featureMap = new HashMap<>();
      for (String featureName : featureList) {
        featureMap.put(familyName, featureName);
      }
      transform.featureNames(featureMap);
    }
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
      for (Map.Entry<String, String> pair : featureNames.entrySet()) {
        features.add(registry.feature(pair.getKey(), pair.getValue()));
      }
    }
  }

  @Override
  protected void validate() {
    super.validate();
    if (familyNames == null && featureNames == null) {
      // TODO (Brad): Maybe this should be a constraint violation exception.
      throw new IllegalArgumentException( "At least one of familyNames or featureNames must be set.");
    }
  }

  @Override
  protected void doTransform(MultiFamilyVector vector) {
    if (features != null) {
      for (Feature feature : features) {
        vector.removeDouble(feature);
      }
    }

    if (families != null) {
      for (Family family : families) {
        vector.remove(family);
      }
    }
  }
}
