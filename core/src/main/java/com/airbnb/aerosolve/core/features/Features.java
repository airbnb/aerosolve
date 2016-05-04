package com.airbnb.aerosolve.core.features;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.google.common.annotations.VisibleForTesting;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;

import java.lang.reflect.Field;
import java.util.*;

public class Features {
  public final static String DEFAULT_STRING_FAMILY = "L";
  public final static String NO_DEFAULT_STRING_FAMILY = "";
  public final static String LABEL = "LABEL";
  public final static String LABEL_FEATURE_NAME = "";
  public final static String MISS = "MISS";
  // In RAW case, don't append feature name
  public final static String RAW = "RAW";
  private final static char FAMILY_SEPARATOR = '_';
  private final static char TRUE_FEATURE = 'T';
  private final static char FALSE_FEATURE = 'F';

  public final String[] names;
  public final Object[] values;
  private final String defaultStringFamilyName;

  public static Features genDefaultStringFamilyFeatures(String[] names, Object[] values) {
    return new Features(names, values);
  }

  public static Features genFeaturesWithCustomerStringFamily(
      String[] names, Object[] values, String defaultStringFamilyName) {
    return new Features(names, values, defaultStringFamilyName);
  }

  public static Features genFeaturesWithoutDefaultStringFamily(
      String[] names, Object[] values) {
    return new Features(names, values, NO_DEFAULT_STRING_FAMILY);
  }

  /*
    if all names has family names in the prefix
    pass defaultStringFamilyName = "" or null
   */
  private Features(String[] names, Object[] values, String defaultStringFamilyName) {
    this.names = names;
    this.values = values;
    this.defaultStringFamilyName = defaultStringFamilyName;
  }

  private Features(String[] names, Object[] values) {
    this(names, values, DEFAULT_STRING_FAMILY);
  }

  private boolean hasDefaultStringFamily() {
    return !defaultStringFamilyName.isEmpty();
  }

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
    assert (names.length == values.length);
    if (names.length != values.length) {
      throw new RuntimeException("names.length != values.length");
    }
    Example example = new Example();
    FeatureVector featureVector = new FeatureVector();
    example.addToExample(featureVector);

    // Set string features.
    final Map<String, Set<String>> stringFeatures = new HashMap<>();
    featureVector.setStringFeatures(stringFeatures);

    final Map<String, Map<String, Double>> floatFeatures = new HashMap<>();
    featureVector.setFloatFeatures(floatFeatures);

    final Set<String> bias = new HashSet<>();
    final Set<String> missing = new HashSet<>();

    bias.add("B");
    stringFeatures.put("BIAS", bias);
    stringFeatures.put(MISS, missing);
    Set<String> defaultStringFamily = null;
    if (hasDefaultStringFamily()) {
      defaultStringFamily = new HashSet<>();
      stringFeatures.put(DEFAULT_STRING_FAMILY, defaultStringFamily);
    }

    for (int i = 0; i < names.length; i++) {
      String name = names[i];
      Object value = values[i];
      if (value == null) {
        missing.add(name);
      } else if (defaultStringFamily == null){
        Pair<String, String> feature = getFamily(name);
        addFeature(isMultiClass, feature, value, floatFeatures, stringFeatures);
      } else {
        addFeature(isMultiClass, name, value, floatFeatures, defaultStringFamily);
      }
    }
    return example;
  }

  private void addFeature(
      boolean isMultiClass, Pair<String, String> feature, Object value,
      Map<String, Map<String, Double>> floatFeatures,
      final Map<String, Set<String>> stringFeatures) {
    if (value instanceof String) {
      String str = (String) value;
      if (isMultiClass && isLabel(feature.getRight())) {
        addMultiClassLabel(str, floatFeatures);
      } else {
        addStringFeature(str, feature, stringFeatures);
      }
    } else if (value instanceof Boolean) {
      Boolean b = (Boolean) value;
      addBoolFeature(b, feature, stringFeatures);
    } else {
      addNumberFeature((Number) value, feature, floatFeatures);
    }
  }

  private void addFeature(
      boolean isMultiClass, String name, Object value,
      Map<String, Map<String, Double>> floatFeatures,
      Set<String> defaultStringFamily) {
    if (value instanceof String) {
      String str = (String) value;
      if (isMultiClass && isLabel(name)) {
        addMultiClassLabel(str, floatFeatures);
      } else {
        addStringFeature(str, name, defaultStringFamily);
      }
    } else if (value instanceof Boolean) {
      Boolean b = (Boolean) value;
      addBoolFeature(b, name, defaultStringFamily);
    } else {
      Pair<String, String> feature = getFamily(name);
      addNumberFeature((Number) value, feature, floatFeatures);
    }
  }

  static void addBoolFeature(Boolean b, String name, Set<String> feature) {
    char str = (b.booleanValue()) ? TRUE_FEATURE : FALSE_FEATURE;
    feature.add(name + ':' + str);
  }

  @VisibleForTesting
  static void addNumberFeature(
      Number value, Pair<String, String> featurePair, Map<String, Map<String, Double>> floatFeatures) {
    Map<String, Double> feature = Util.getOrCreateFloatFeature(featurePair.getLeft(), floatFeatures);
    feature.put(featurePair.getRight(), value.doubleValue());
  }

  @VisibleForTesting
  static void addBoolFeature(
      Boolean b, Pair<String, String> featurePair, Map<String, Set<String>> stringFeatures) {
    Set<String> feature = Util.getOrCreateStringFeature(featurePair.getLeft(), stringFeatures);
    String featureName = featurePair.getRight();
    addBoolFeature(b, featureName, feature);
  }

  @VisibleForTesting
  static void addStringFeature(
      String value, Pair<String, String> featurePair, Map<String, Set<String>> stringFeatures) {
    Set<String> feature = Util.getOrCreateStringFeature(featurePair.getLeft(), stringFeatures);
    addStringFeature(value, featurePair.getRight(), feature);
  }

  static void addStringFeature(
      String value, String featureName, Set<String> feature) {
    if (featureName.equals(RAW)) {
      feature.add(value);
    } else {
      feature.add(featureName + ":" + value);
    }
  }


  @VisibleForTesting
  static void addMultiClassLabel(String str, Map<String, Map<String, Double>> floatFeatures) {
    String[] labels =  str.split(",");
    for (String s: labels) {
      String[] labelTokens = s.split(":");
      if (labelTokens.length != 2) {
        throw new RuntimeException(String.format(
            "MultiClass LABEL \"%s\" not in format [label1]:[weight1],...!", str));
      }
      Map<String, Double> feature = Util.getOrCreateFloatFeature(LABEL, floatFeatures);
      feature.put(labelTokens[0], Double.valueOf(labelTokens[1]));
    }
  }

  static boolean isLabel(Pair<String, String> featurePair) {
    return featurePair.getRight().equals(LABEL_FEATURE_NAME);
  }

  static boolean isLabel(String name) {
    return name.equals(LABEL_FEATURE_NAME);
  }

  @VisibleForTesting
  static Pair<String, String> getFamily(String name) {
    int pos = name.indexOf(FAMILY_SEPARATOR);
    if (pos == -1) {
      if (name.compareToIgnoreCase(LABEL) == 0) {
        return new ImmutablePair<>(LABEL, LABEL_FEATURE_NAME) ;
      } else {
        throw new RuntimeException("Column name not in FAMILY_NAME format or is not LABEL! " + name);
      }
    } else if (pos == 0) {
      throw new RuntimeException("Column name can't prefix with _! " + name);
    } else {
      return new ImmutablePair<>(name.substring(0, pos),
          name.substring(pos + 1));
    }
  }
}
