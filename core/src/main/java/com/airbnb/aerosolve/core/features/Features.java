package com.airbnb.aerosolve.core.features;

import lombok.experimental.Builder;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Builder
public class Features {
  public static final Integer StringType = 0;
  // float type includes int, long, float, double
  public static final Integer FloatType = 1;
  public static final Integer BooleanType = 2;
  public static final Integer TypeSize = 3;

  public final String[] names;
  public final Object[] values;
  public final Integer[] types;

  /*
    Util function to get features for FeatureMapping
   */
  public static List<ScoringFeature> getGenericSortedFeatures(Class c) {
    return getGenericSortedFeatures(c.getDeclaredFields());
  }

  public static List<ScoringFeature> getGenericSortedFeatures(Field[] fields) {
    List<ScoringFeature> features = new ArrayList<>();

    for (Field field : fields) {
      Class c = field.getType();
      if(c.equals(String.class)) {
        features.add(new ScoringFeature(field.getName(), StringType));
      } else if (c.equals(Double.class)) {
        features.add(new ScoringFeature(field.getName(), FloatType));
      } else if (c.equals(Boolean.class)) {
        features.add(new ScoringFeature(field.getName(), BooleanType));
      }
    }
    // Sort the non-amenity features alphabetically
    Collections.sort(features);
    return features;
  }
}
