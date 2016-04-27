package com.airbnb.aerosolve.core.features;

import lombok.experimental.Builder;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Builder
public class Features {
  public final String[] names;
  public final Object[] values;

  /*
    Util function to get features for FeatureMapping
   */
  public static List<String> getGenericSortedFeatures(Class c) {
    return getGenericSortedFeatures(c.getDeclaredFields());
  }

  public static List<String> getGenericSortedFeatures(Field[] fields) {
    List<String> features = new ArrayList<>();

    for (Field field : fields) {
      features.add(field.getName());
    }
    // Sort the non-amenity features alphabetically
    Collections.sort(features);
    return features;
  }
}
