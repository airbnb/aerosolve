package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;

import com.typesafe.config.Config;
import java.util.HashSet;
import java.util.Set;
import java.util.Map;

/**
 * Takes the self cross product of stringFeatures named in field1
 * and places it in a stringFeature with family name specified in output.
 */
public class SelfCrossTransform implements Transform {
  private String fieldName1;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    outputName = config.getString(key + ".output");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    if (stringFeatures == null) return;

    Set<String> set1 = stringFeatures.get(fieldName1);
    if (set1 == null) return;

    Set<String> output = Util.getOrCreateStringFeature(outputName, stringFeatures);

    selfCross(set1, output);
  }

  public static void selfCross(Set<String> set1, Set<String> output) {
    for (String s1 : set1) {
      for (String s2 : set1) {
        // To prevent duplication we only take pairs there s1 < s2.
        if (s1.compareTo(s2) < 0) {
          output.add(s1 + '^' + s2);
        }
      }
    }
  }
}
