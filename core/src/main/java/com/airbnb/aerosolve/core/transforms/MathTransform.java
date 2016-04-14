package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.BaseFeaturesTransform;
import com.google.common.base.Optional;
import com.typesafe.config.Config;
import java.util.function.DoubleUnaryOperator;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;

import javax.validation.constraints.NotNull;

/**
 * Apply given Math function on specified float features defined by fieldName1 and keys
 * fieldName1: feature family name
 * keys: feature names
 * outputName: output feature family name (feature names or keys remain the same)
 * function: a string that specified the function that is going to apply to the given feature
 */
@LegacyNames("math_float")
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class MathTransform extends BaseFeaturesTransform<MathTransform> {
  @NotNull
  private String functionName;

  @Setter(AccessLevel.NONE)
  private Optional<DoubleUnaryOperator> func;

  @Override
  public MathTransform configure(Config config, String key) {
    return super.configure(config, key)
        .functionName(stringFromConfig(config, key, ".function"));
  }

  @Override
  protected void setup() {
    super.setup();
    func = getFunction(functionName);
    if (!func.isPresent()) {
      throw new IllegalArgumentException(
          String.format("Cannot run math transform. %s function is unknown.", functionName));
    }
  }

  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    for (FeatureValue value : getInput(featureVector)) {
      if (featureVector.containsKey(value.feature())) {
        double v = value.value();
        double result = func.get().applyAsDouble(v);
        if (Double.isNaN(result)
            || Double.isInfinite(result)) {
          continue;
        }
        Feature outputFeature = produceOutputFeature(value.feature());
        featureVector.put(outputFeature, result);
      }
    }
  }

  private Optional<DoubleUnaryOperator> getFunction(String functionName) {
    switch (functionName) {
      case "sin":
        return Optional.of(Math::sin);
      case "cos":
        return Optional.of(Math::cos);
      case "log10":
        // return the original value if x <= 0
        return Optional.of(Math::log10);
      case "log":
        // return the original value if x <= 0
        return Optional.of(Math::log);
      case "abs":
        return Optional.of(Math::abs);
    }
    return Optional.absent();
  }
}
