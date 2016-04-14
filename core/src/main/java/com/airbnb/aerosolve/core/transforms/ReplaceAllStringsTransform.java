package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.transforms.types.StringTransform;
import com.typesafe.config.Config;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Setter;
import lombok.experimental.Accessors;
import org.apache.commons.lang3.tuple.Pair;
import org.hibernate.validator.constraints.NotEmpty;

import javax.validation.constraints.NotNull;

/**
 * Replaces all substrings that match a given regex with a replacement string
 * "field1" specifies the key of the feature
 * "replacements" specifies a list of pairs (or maps) of regexes and corresponding replacements
 * Replacements are performed in the same order as specified in the list of pairs
 * "replacement" specifies the replacement string
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
public class ReplaceAllStringsTransform extends StringTransform<ReplaceAllStringsTransform> {
  @NotNull
  @NotEmpty
  private Map<String, String> replacements;

  @Setter(AccessLevel.NONE)
  private List<Pair<Pattern, String>> patterns;

  @Override
  public ReplaceAllStringsTransform configure(Config config, String key) {
    return super.configure(config, key)
        .replacements(stringMapFromConfig(config, key, ".replacements", true));
  }

  @Override
  protected void setup() {
    super.setup();
    patterns = replacements.entrySet().stream()
        .map(e -> Pair.of(Pattern.compile(e.getKey()), e.getValue()))
        .collect(Collectors.toList());
  }

  @Override
  public String processString(String rawString) {
    for (Pair<Pattern, String> replacement : patterns) {
      rawString = replacement.getKey().matcher(rawString).replaceAll(replacement.getValue());
    }
    return rawString;
  }
}
