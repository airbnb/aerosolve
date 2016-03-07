package com.airbnb.aerosolve.core.online;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static com.sun.tools.internal.xjc.reader.Ring.add;

public abstract class ExampleGen {
  public Example gen() {
    Example example = new Example();
    FeatureVector featureVector = new FeatureVector();
    example.addToExample(featureVector);

    // Set string features.
    final Map<String, Set<String>> stringFeatures = new HashMap<>();
    featureVector.setStringFeatures(stringFeatures);
    setBIAS(stringFeatures);
    final Set<String> stringFamily = addStringFamily(getDefaultStringFamilyName(), stringFeatures);

    final Map<String, Map<String, Double>> floatFeatures = new HashMap<>();
    featureVector.setFloatFeatures(floatFeatures);

    addStringFamily(stringFeatures);
    return example;
  }

  protected Set<String> addStringFamily(final String name, final Map<String, Set<String>> stringFeatures) {
    final Set<String> stringFamily = new HashSet<>();
    stringFeatures.put(name, stringFamily);
    return stringFamily;
  }

  protected void setBIAS(final Map<String, Set<String>> stringFeatures) {
    addStringFamily("BIAS", stringFeatures).add("B");
  }

  protected String getDefaultStringFamilyName() {
    return "S";
  }

}
