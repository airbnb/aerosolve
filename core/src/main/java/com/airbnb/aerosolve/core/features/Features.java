package com.airbnb.aerosolve.core.features;


import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.perf.Family;
import com.airbnb.aerosolve.core.perf.Feature;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.airbnb.aerosolve.core.perf.SimpleExample;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import lombok.experimental.Builder;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;

@Builder @Slf4j
public class Features {
  public final static String LABEL = "LABEL";
  public final static String LABEL_FEATURE_NAME = "";
  public final static String MISS = "MISS";
  // In RAW case, don't append feature name
  public final static String RAW = "RAW";
  private final static char FAMILY_SEPARATOR = '_';
  private final static char TRUE_FEATURE = 'T';
  private final static char FALSE_FEATURE = 'F';
  public static final char NAME_SEPARATOR = ':';

  public final String[] names;
  public final Object[] values;
  private final FeatureRegistry registry;

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

  // TODO  make it more generic, for example, taking care of dense feature
  public Example toExample(boolean isMultiClass) {
    Preconditions.checkState(names.length == values.length, "names.length != values.length");
    Example example = new SimpleExample(registry);
    MultiFamilyVector featureVector = example.createVector();

    featureVector.putString(registry.feature("BIAS", "B"));
    Family missFamily = registry.family(MISS);
    Family labelFamily = registry.family(LABEL);

    for (int i = 0; i < names.length; i++) {
      String name = names[i];
      Object value = values[i];
      if (value == null) {
        featureVector.putString(missFamily.feature(name));
      } else {
        Feature feature = calculateFeature(name, value, registry);
        if (isMultiClass && feature.family() == labelFamily) {
          addMultiClassLabel((String) value, featureVector, labelFamily);
        } else if (value instanceof Number) {
          featureVector.put(feature, ((Number)value).doubleValue());
        } else {
          featureVector.putString(feature);
        }
      }
    }
    return example;
  }

  @VisibleForTesting
  static void addMultiClassLabel(String str, FeatureVector vector, Family labelFamily) {
    String[] labels =  str.split(",");
    for (String s: labels) {
      String[] labelTokens = s.split(":");
      if (labelTokens.length != 2) {
        throw new RuntimeException(String.format(
            "MultiClass LABEL \"%s\" not in format [label1]:[weight1],...!", str));
      }
      vector.put(labelFamily.feature(labelTokens[0]), Double.valueOf(labelTokens[1]));
    }
  }

  static boolean isLabel(Pair<String, String> feature) {
    return feature.getLeft().equals(LABEL) && feature.getRight().equals(LABEL_FEATURE_NAME);
  }

  @VisibleForTesting
  static Feature calculateFeature(String name, Object value, FeatureRegistry registry) {
    if (name.equals(LABEL)) {
      return registry.feature(LABEL, LABEL_FEATURE_NAME);
    }
    int pos = name.indexOf(FAMILY_SEPARATOR);
    if (pos <= 0) {
      throw new RuntimeException(
          String.format("Column name %s is invalid. It must either be %s or start with a family "
                        + "name followed by %s and a feature name.", name, LABEL, FAMILY_SEPARATOR));
    }
    String familyName = name.substring(0, pos);
    String featureName = name.substring(pos + 1);
    if (value instanceof String && !featureName.equals(RAW)) {
      featureName = featureName + NAME_SEPARATOR + value;
    } else if (value instanceof Boolean) {
      featureName = featureName + NAME_SEPARATOR + ((boolean)value ? TRUE_FEATURE : FALSE_FEATURE);
    }
    return registry.feature(familyName, featureName);
  }
}
