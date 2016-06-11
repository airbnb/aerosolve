package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;
import lombok.extern.slf4j.Slf4j;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Created by hector_yee on 8/25/14.
 * Takes the cross product of stringFeatures named in field1 and field2
 * and places it in a stringFeature with family name specified in output.
 */
@Slf4j
public class CrossTransform implements Transform {
  private String fieldName1;
  private String fieldName2;
  private String outputName;
  private Set<String> keys1;
  private Set<String> keys2;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    fieldName2 = config.getString(key + ".field2");
    outputName = config.getString(key + ".output");
    String key1Name = key + ".keys1";
    String key2Name = key + ".keys2";
    if (config.hasPath(key1Name)) {
      keys1 = new HashSet<>(config.getStringList(key1Name));
    }
    if (config.hasPath(key2Name)) {
      keys2 = new HashSet<>(config.getStringList(key2Name));
    }
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    if (stringFeatures == null) return;

    Set<String> set1 = stringFeatures.get(fieldName1);
    if (set1 == null || set1.isEmpty()) return;
    Set<String> set2 = stringFeatures.get(fieldName2);
    if (set2 == null || set2.isEmpty()) return;

    Set<String> output = stringFeatures.get(outputName);
    if (output == null) {
      output = new HashSet<>();
      stringFeatures.put(outputName, output);
    }

    Set<String> localKeys1 = (keys1 == null) ? set1 : Util.getIntersection(keys1, set1);
    if (localKeys1.isEmpty()) return;
    Set<String> localKeys2 = (keys2 == null) ? set2 : Util.getIntersection(keys2, set2);
    if (localKeys2.isEmpty()) return;

    cross(localKeys1, localKeys2, output);
  }

  public static void cross(Set<String> set1, Set<String> set2, Set<String> output) {
    for (String s1 : set1) {
      String prefix = s1 + '^';
      for (String s2 : set2) {
        output.add(prefix + s2);
      }
    }
  }
 }
