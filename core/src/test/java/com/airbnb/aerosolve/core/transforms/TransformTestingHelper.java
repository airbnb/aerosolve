package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class TransformTestingHelper {
  public static FeatureVector makeFeatureVector() {
    Map<String, Set<String>> stringFeatures = new HashMap<>();
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Set list = new HashSet<String>();
    list.add("aaa");
    list.add("bbb");
    stringFeatures.put("strFeature1", list);

    Map<String, Double> map = new HashMap<>();
    map.put("lat", 37.7);
    map.put("long", 40.0);
    map.put("z", -20.0);
    floatFeatures.put("loc", map);

    Map<String, Double> map2 = new HashMap<>();
    map2.put("foo", 1.5);
    floatFeatures.put("F", map2);

    Map<String, Double> map3 = new HashMap<>();
    map3.put("bar_fv", 1.0);
    floatFeatures.put("bar", map3);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }
}
