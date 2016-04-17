package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;

import java.util.List;
import java.util.Map;
import java.util.Set;

import com.typesafe.config.Config;
import com.typesafe.config.ConfigObject;

/**
 * Replaces all substrings that match a given regex with a replacement string
 * "field1" specifies the key of the feature
 * "replacements" specifies a list of pairs (or maps) of regexes and corresponding replacements
 * Replacements are performed in the same order as specified in the list of pairs
 * "replacement" specifies the replacement string
 */
public class ReplaceAllStringsTransform implements Transform {
  private String fieldName1;
  private List<? extends ConfigObject> replacements;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    replacements = config.getObjectList(key + ".replacements");
    outputName = config.getString(key + ".output");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    if (stringFeatures == null) {
      return;
    }

    Set<String> feature1 = stringFeatures.get(fieldName1);
    if (feature1 == null) {
      return;
    }

    Set<String> output = Util.getOrCreateStringFeature(outputName, stringFeatures);

    for (String rawString : feature1) {
      if (rawString == null) continue;
      for (ConfigObject replacementCO : replacements) {
        Map<String, Object> replacementMap = replacementCO.unwrapped();
        for (Map.Entry<String, Object> replacementEntry : replacementMap.entrySet()) {
          String regex = replacementEntry.getKey();
          String replacement = (String) replacementEntry.getValue();
          rawString = rawString.replaceAll(regex, replacement);
        }
      }
      output.add(rawString);
    }
  }
}
