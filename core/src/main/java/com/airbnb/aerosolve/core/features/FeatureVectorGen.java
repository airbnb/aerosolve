package com.airbnb.aerosolve.core.features;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.perf.SimpleExample;
import java.util.List;

/*
  Generate Example from input features and defined featureFamily.
  refer to ModelScorerTest.java as how to use FeatureVectorGen
 */
public class FeatureVectorGen {

  // TODO add a new function to consider dense feature.
  protected static void populateVector(FeatureVector vector,
                                                FeatureRegistry registry,
                                                Features features,
                                                List<StringFamily> stringFamilies,
                                                List<FloatFamily> floatFamilies) {
    // Set string features.
    setBIAS(registry, vector);

    for (int i = 0; i < features.names.length; ++i) {
      Object feature = features.values[i];

      if (feature != null) {
       // Integer type = features.types[i];
        String name = features.names[i];
        if (feature instanceof Double || feature instanceof Float ||
            feature instanceof Integer || feature instanceof Long) {
          for (FloatFamily featureFamily : floatFamilies) {
            if (featureFamily.isMyFamily(name)) {
              vector.put(registry.feature(featureFamily.getFamilyName(), name),
                         (double) feature);
              break;
            }
          }
        } else if (feature instanceof String) {
          for (StringFamily featureFamily : stringFamilies) {
            if (featureFamily.isMyFamily(name)) {
              vector.putString(registry.feature(featureFamily.getFamilyName(),
                                                name + ":" + feature));
              break;
            }
          }
        } else if (feature instanceof Boolean){
          for (StringFamily featureFamily : stringFamilies) {
            if (featureFamily.isMyFamily(name)) {
              vector.putString(registry.feature(featureFamily.getFamilyName(),
                              StringFamily.getBooleanFeatureAsString(name, (Boolean) feature)));
              break;
            }
          }
        }
      }
    }
  }

  public static Example toSingleFeatureVectorExample(Features features,
                                                     List<StringFamily> stringFamilies,
                                                     List<FloatFamily> floatFamilies,
                                                     FeatureRegistry registry) {
    Example example = new SimpleExample(registry);
    FeatureVector vector = example.createVector();
    populateVector(vector, registry, features, stringFamilies, floatFamilies);
    return example;
  }

  protected static void setBIAS(FeatureRegistry registry, FeatureVector vector) {
    vector.putString(registry.feature("BIAS", "B"));
  }
}
