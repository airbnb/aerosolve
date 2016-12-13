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
  // lower case label field will be inserted into Upper case LABEL family name.
  public final static String LABEL = "LABEL";
  public final static String LABEL_FEATURE_NAME = "";
  public final static String MISS = "MISS";

  // for string feature without family name
  public final static String DEFAULT_STRING_FAMILY = "DEFAULT_STRING";
  // for float feature without family name
  public final static String DEFAULT_FLOAT_FAMILY = "DEFAULT_FLOAT";

  // In RAW case, don't append feature name
  public final static String RAW = "RAW";
  private final static char FAMILY_SEPARATOR = '_';
  private final static char TRUE_FEATURE = 'T';
  private final static char FALSE_FEATURE = 'F';
  private final static String STRING_FEATURE_SEPARATOR= ":";
  public final String[] names;
  public final Object[] values;

  // names starting with _meta_ will be treated as metadata instead of features
  private final static String METADATA_PREFIX = "_meta_";

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
    Example example = new Example();
    FeatureVector featureVector = new FeatureVector();
    example.addToExample(featureVector);

    // Set string features.
    final Map<String, Set<String>> stringFeatures = new HashMap<>();
    featureVector.setStringFeatures(stringFeatures);

    final Map<String, Map<String, Double>> floatFeatures = new HashMap<>();
    featureVector.setFloatFeatures(floatFeatures);
    // create LABEL family
    floatFeatures.put(LABEL, new HashMap<>());

    final Set<String> bias = new HashSet<>();
    final Set<String> missing = new HashSet<>();
    bias.add("B");
    stringFeatures.put("BIAS", bias);
    stringFeatures.put(MISS, missing);

    // metadata map
    final Map<String, String> metadata = new HashMap<>();
    example.setMetadata(metadata);

    for (int i = 0; i < names.length; i++) {
      String name = names[i];
      Object value = values[i];
      if (isMetadata(name)) {
        metadata.put(name.substring(METADATA_PREFIX.length()), value == null ? null : value.toString());
      } else {
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
    }
    return example;
  }

  private static boolean isMetadata(String name) {
    return name.startsWith(METADATA_PREFIX);
  }

  @VisibleForTesting
  static void addNumberFeature(
      Number value, Pair<String, String> featurePair, Map<String, Map<String, Double>> floatFeatures) {
    String family = getFloatFamily(featurePair);
    Map<String, Double> feature = Util.getOrCreateFloatFeature(family, floatFeatures);
    feature.put(featurePair.getRight(), value.doubleValue());
  }

  static String getFloatFamily(Pair<String, String> featurePair) {
    String left = featurePair.getLeft();
    return left.isEmpty()? DEFAULT_FLOAT_FAMILY :left;
  }

  static String getStringFamily(Pair<String, String> featurePair) {
    String left = featurePair.getLeft();
    return left.isEmpty()? DEFAULT_STRING_FAMILY :left;
  }

  @VisibleForTesting
  static void addBoolFeature(
      Boolean b, Pair<String, String> featurePair, Map<String, Set<String>> stringFeatures) {
    String family = getStringFamily(featurePair);
    Set<String> feature = Util.getOrCreateStringFeature(family, stringFeatures);
    String featureName = featurePair.getRight();
    char str = b ? TRUE_FEATURE : FALSE_FEATURE;
    feature.add(featureName + STRING_FEATURE_SEPARATOR + str);
  }

  @VisibleForTesting
  static void addStringFeature(
      String str, Pair<String, String> featurePair, Map<String, Set<String>> stringFeatures) {
    String family = getStringFamily(featurePair);
    Set<String> feature = Util.getOrCreateStringFeature(family, stringFeatures);
    String featureName = featurePair.getRight();
    if (featureName.equals(RAW)) {
      feature.add(str);
    } else {
      feature.add(featureName + STRING_FEATURE_SEPARATOR + str);
    }
  }

  // string feature is concatenated by : the prefix before : is feature name
  // RAW feature has no : so just return the RAW
  // this is used in StringCrossFloatTransform so that
  // we can cross Raw feature as well as other string features
  public static String getStringFeatureName(String feature) {
    String[] tokens =  feature.split(STRING_FEATURE_SEPARATOR);
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
      if (name.compareToIgnoreCase(LABEL) == 0) {
        return new ImmutablePair<>(LABEL, LABEL_FEATURE_NAME) ;
      } else if (!name.isEmpty()){
        return new ImmutablePair<>("", name) ;
      } else {
        throw new RuntimeException("Column name empty");
      }
    } else if (pos == 0) {
      throw new RuntimeException("Column name can't prefix with _! " + name);
    } else {
      return new ImmutablePair<>(name.substring(0, pos),
          name.substring(pos + 1));
    }
  }
}
