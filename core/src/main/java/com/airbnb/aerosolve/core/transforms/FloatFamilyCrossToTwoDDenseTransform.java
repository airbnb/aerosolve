package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * cross one float family with the other and out a 2D dense feature.
 * if fieldsName2 missing, this is a self cross
 * self cross use key's alphabetical order to determine order
 * so you can add 1_ 2_ in front of features to manipulate order.
 */
public class FloatFamilyCrossToTwoDDenseTransform implements Transform {
  private String fieldName1;
  private String fieldName2;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    String path = key + ".field2";
    if (config.hasPath(path)) {
      fieldName2 = config.getString(path);
    }
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();

    if (floatFeatures == null) {
      return;
    }

    Map<String, Double> map1 = floatFeatures.get(fieldName1);
    if (map1 == null || map1.isEmpty()) return;

    if (fieldName2 != null) {
      Map<String, Double> map2 = floatFeatures.get(fieldName2);
      if (map2 == null || map2.isEmpty()) return;

      cross(map1, map2, featureVector);
    } else {
      selfCross(map1, featureVector);
    }
  }

  private void selfCross(Map<String, Double> map1, FeatureVector featureVector) {
    if (map1.size() <= 1) return;
    List<Map.Entry<String, Double>> list = new ArrayList<>(map1.size());
    list.addAll(map1.entrySet());
    for (int i = 0; i < list.size(); ++i) {
      for (int j = i+1; j < list.size(); ++j) {
        Map.Entry<String, Double> a = list.get(i);
        Map.Entry<String, Double> b = list.get(j);
        String key1 = a.getKey();
        String key2 = b.getKey();
        // use key's alphabetical order to determine order
        if (key1.compareTo(key2) < 0) {
          Util.setDenseFeature(featureVector,
              a.getKey() + "^" + b.getKey(),
              Arrays.asList(a.getValue(), b.getValue()));
        } else {
          Util.setDenseFeature(featureVector,
              b.getKey() + "^" + a.getKey(),
              Arrays.asList(b.getValue(), a.getValue()));
        }
      }
    }
  }

  static void cross(
      Map<String, Double> map1,
      Map<String, Double> map2,
      FeatureVector featureVector) {
    for (Map.Entry<String, Double> a : map1.entrySet()) {
      for (Map.Entry<String, Double> b : map2.entrySet()) {
        Util.setDenseFeature(featureVector,
            a.getKey() + "^" + b.getKey(),
            Arrays.asList(a.getValue(), b.getValue()));
      }
    }
  }
}
