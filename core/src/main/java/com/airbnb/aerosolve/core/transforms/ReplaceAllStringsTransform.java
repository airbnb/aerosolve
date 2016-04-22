package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.transforms.types.StringTransform;

import java.util.List;
import java.util.Map;

import com.typesafe.config.Config;
import com.typesafe.config.ConfigObject;

/**
 * Replaces all substrings that match a given regex with a replacement string
 * "field1" specifies the key of the feature
 * "replacements" specifies a list of pairs (or maps) of regexes and corresponding replacements
 * Replacements are performed in the same order as specified in the list of pairs
 * "replacement" specifies the replacement string
 */
public class ReplaceAllStringsTransform extends StringTransform {
  private List<? extends ConfigObject> replacements;

  @Override
  public void init(Config config, String key) {
    replacements = config.getObjectList(key + ".replacements");
  }

  @Override
  public String processString(String rawString) {
    if (rawString == null) {
      return null;
    }

    for (ConfigObject replacementCO : replacements) {
      Map<String, Object> replacementMap = replacementCO.unwrapped();

      for (Map.Entry<String, Object> replacementEntry : replacementMap.entrySet()) {
        String regex = replacementEntry.getKey();
        String replacement = (String) replacementEntry.getValue();
        rawString = rawString.replaceAll(regex, replacement);
      }
    }

    return rawString;
  }
}
