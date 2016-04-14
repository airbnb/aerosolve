package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.transforms.types.StringTransform;
import com.typesafe.config.Config;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

/**
 * Converts strings to either all lowercase or all uppercase
 * "field1" specifies the key of the feature
 * "convert_to_uppercase" converts strings to uppercase if true, otherwise converts to lowercase
 * "output" optionally specifies the key of the output feature, if it is not given the transform
 * overwrites / replaces the input feature
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
public class ConvertStringCaseTransform extends StringTransform<ConvertStringCaseTransform> {
  protected boolean convertToUppercase;

  public ConvertStringCaseTransform configure(Config config, String key) {
    return super.configure(config, key)
        .convertToUppercase(booleanFromConfig(config, key, ".convert_to_uppercase"));
  }

  @Override
  public String processString(String rawString) {
    return convertToUppercase ? rawString.toUpperCase() : rawString.toLowerCase();
  }
}
