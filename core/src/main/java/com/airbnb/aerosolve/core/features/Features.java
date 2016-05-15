package com.airbnb.aerosolve.core.features;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.google.common.annotations.VisibleForTesting;
import lombok.experimental.Builder;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;

import java.lang.reflect.Field;
import java.util.*;

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

    for (int i = 0; i < names.length; i++) {
      String name = names[i];
      Object value = values[i];
      if (value == null) {
        missing.add(name);
      } else {
        Pair<String, String> feature = getFamily(name);
        if (value instanceof String) {
          String str = (String) value;
          if (isMultiClass && isLabel(feature)) {
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
    }
    return example;
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
    char str = (b.booleanValue()) ? TRUE_FEATURE : FALSE_FEATURE;
    feature.add(featureName + ':' + str);
  }

  @VisibleForTesting
  static void addStringFeature(
      String str, Pair<String, String> featurePair, Map<String, Set<String>> stringFeatures) {
    Set<String> feature = Util.getOrCreateStringFeature(featurePair.getLeft(), stringFeatures);
    String featureName = featurePair.getRight();
    if (featureName.equals(RAW)) {
      feature.add(str);
    } else {
      feature.add(featureName + ":" + str);
    }
  }

  // string feature is concatenated by : the prefix before : is feature name
  // RAW feature has no : so just return the RAW
  // this is used in StringCrossFloatTransform so that
  // we can cross Raw feature as well as other string features
  public static String getStringFeatureName(String feature) {
    String[] tokens =  feature.split(":");
    if (tokens.length == 1) {
      return RAW;
    } else {
      return tokens[0];
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

  static boolean isLabel(Pair<String, String> feature) {
    return feature.getRight().equals(LABEL_FEATURE_NAME);
  }

  @VisibleForTesting
  static Pair<String, String> getFamily(String name) {
    int pos = name.indexOf(FAMILY_SEPARATOR);
    if (pos == -1) {
      if (name.compareTo(LABEL) == 0) {
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
