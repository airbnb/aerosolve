package com.airbnb.aerosolve.core.features;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;

import java.util.*;

/*
  Generate Example from input features and defined featureFamily.
  refer to ModelScorerTest.java as how to use ExampleGen
 */
public class ExampleGen {

  public static Example toExample(Features features,
                           List<StringFamily> stringFamilies,
                           List<FloatFamily> floatFamilies) {
    Example example = new Example();
    FeatureVector featureVector = new FeatureVector();
    example.addToExample(featureVector);

    // Set string features.
    final Map<String, Set<String>> stringFeatures = new HashMap<>();
    featureVector.setStringFeatures(stringFeatures);
    setBIAS(stringFeatures);

    for (StringFamily featureFamily : stringFamilies) {
      stringFeatures.put(featureFamily.getFamilyName(), featureFamily.getFeatures());
    }

    final Map<String, Map<String, Double>> floatFeatures = new HashMap<>();
    featureVector.setFloatFeatures(floatFeatures);
    for (FloatFamily featureFamily : floatFamilies) {
      floatFeatures.put(featureFamily.getFamilyName(), featureFamily.getFeatures());
    }

    for (int i = 0; i < features.names.length; ++i) {
      Object feature = features.values[i];

      if (feature != null) {
        Integer type = features.types[i];
        String name = features.names[i];
        if (type == Features.FloatType) {
          for (FloatFamily featureFamily : floatFamilies) {
            if (featureFamily.add(name, feature)) break;
          }
        } else {
          assert(type == Features.StringType || type == Features.BooleanType);
          for (StringFamily featureFamily : stringFamilies) {
            if (featureFamily.add(name, feature, type)) break;
          }
        }
      }
    }

    return example;
  }

  protected static void setBIAS(final Map<String, Set<String>> stringFeatures) {
    final Set<String> bias = new HashSet<>();
    bias.add("B");
    stringFeatures.put("BIAS", bias);
  }
}
