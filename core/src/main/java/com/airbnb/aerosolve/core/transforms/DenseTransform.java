package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.features.SimpleFeatureValue;
import com.airbnb.aerosolve.core.transforms.base.ConfigurableTransform;
import com.typesafe.config.Config;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;
import org.hibernate.validator.constraints.NotEmpty;

import javax.validation.constraints.NotNull;

/*
  Turn several float features into one dense feature, feature number must > 1
  1. IF all float features are null, create a string feature,
      with family name string_output, feature name output^null
  2. IF only one float feature is not null, create a float feature
     with family name same as family of the only not null float feature
  3. Other cases create dense features
   both 2 and 3, feature name: output^key keys.
 */
@LegacyNames("float_to_dense")
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class DenseTransform extends ConfigurableTransform<DenseTransform> {
  private static final int FEATURE_AVG_SIZE = 16;

  @NotNull
  @NotEmpty
  private List<String> fields;
  @NotNull
  @NotEmpty
  private List<String> keys;
  @NotNull
  private String outputFamilyName;
  @NotNull
  private String outputStringFamilyName;

  @Setter(AccessLevel.NONE)
  private List<Feature> inputFeatures;
  @Setter(AccessLevel.NONE)
  private Family outputFamily;
  @Setter(AccessLevel.NONE)
  private Family outputStringFamily;

  @Override
  public DenseTransform configure(Config config, String key) {
    return outputStringFamilyName(stringFromConfig(config, key, ".string_output"))
        .outputFamilyName(stringFromConfig(config, key, ".output"))
        .keys(stringListFromConfig(config, key, ".keys", true))
        .fields(stringListFromConfig(config, key, ".fields", true));
  }

  @Override
  protected void setup() {
    super.setup();
    outputFamily = registry.family(outputFamilyName);
    outputStringFamily = registry.family(outputStringFamilyName);
    inputFeatures = new ArrayList<>();
    for (int i = 0; i < fields.size(); i++) {
      String familyName = fields.get(i);
      String featureName = keys.get(i);
      inputFeatures.add(registry.feature(familyName, featureName));
    }
  }

  @Override
  protected void validate() {
    super.validate();
    if (fields.size() != keys.size() || fields.size() <= 1) {
      String msg = String.format("fields size {} keys size {}", fields.size(), keys.size());
      throw new RuntimeException(msg);
    }
  }

  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    List<FeatureValue> values = inputFeatures.stream()
        .filter(featureVector::containsKey)
        .map((Feature feature) ->
                 SimpleFeatureValue.of(feature, featureVector.getDouble(feature)))
        .collect(Collectors.toList());

    switch (values.size()) {
      case 0:
        Feature outputFeature = outputStringFamily.feature(outputFamilyName + "^null");
        featureVector.putString(outputFeature);
        break;
      case 1:
        FeatureValue value = values.get(0);
        String name = outputFamilyName + "^" + value.feature().name();
        featureVector.put(outputFamily.feature(name), value.value());
        break;
      default:
        double[] output = new double[values.size()];
        StringBuilder nameBuilder = new StringBuilder();
        for (int i = 0; i < values.size(); i++) {
          output[i] = values.get(i).value();
          nameBuilder.append("^");
          nameBuilder.append(values.get(i).feature().name());
        }
        featureVector.putDense(registry.family(nameBuilder.toString()), output);
    }
  }
}
