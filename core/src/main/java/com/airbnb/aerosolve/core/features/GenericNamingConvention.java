package com.airbnb.aerosolve.core.features;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import lombok.Synchronized;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/**
 *
 */
public class GenericNamingConvention implements NamingConvention, Serializable {

  public final static String LABEL = "LABEL";
  public final static String LABEL_FEATURE_NAME = "";
  public final static String MISS = "MISS";
  // In RAW case, don't append feature name
  public final static String RAW = "RAW";
  private final static char FAMILY_SEPARATOR = '_';
  private final static char TRUE_FEATURE = 'T';
  private final static char FALSE_FEATURE = 'F';
  public static final char NAME_SEPARATOR = ':';

  public static GenericNamingConvention INSTANCE;

  private final Function<String, Map<String, Double>> labelSplitFunction;

  public GenericNamingConvention(
      Function<String, Map<String, Double>> labelSplitFunction) {
    this.labelSplitFunction = labelSplitFunction;
  }

  public static GenericNamingConvention instance() {
    if (INSTANCE == null) {
      setInstance();
    }
    return INSTANCE;
  }

  @Synchronized
  private static void setInstance() {
    INSTANCE = new GenericNamingConvention(GenericNamingConvention::genericLabelSplitFunction);
  }

  @Override
  public NamingConventionResult features(String name, Object value, FeatureRegistry registry) {
    Preconditions.checkNotNull(name, "Cannot create a feature from a null name");
    Feature feature;
    if (value == null) {
      feature = registry.feature(MISS, name);
    } else if (value instanceof double[]) {
      return NamingConventionResult.builder()
          .denseFeatures(ImmutableMap.of(registry.family(name), (double[]) value))
          .build();
    } else if (name.equals(LABEL)) {
      if (value instanceof String) {
        Set<FeatureValue> values = extractFeatureValuesFromLabel((String) value, registry);
        return NamingConventionResult.builder()
            .doubleFeatures(values)
            .build();
      }
      feature = registry.feature(LABEL, LABEL_FEATURE_NAME);
    } else {
      feature = parseFeature(name, value, registry);
    }
    checkState(feature != null, "Could not find a way to parse feature name %s for value %s",
               name, value);
    return createNamingConventionResult(feature, value);
  }

  private Set<FeatureValue> extractFeatureValuesFromLabel(String value, FeatureRegistry registry) {
    // TODO(Brad): This could be a bit of a problem if a non-multiclass feature had a String
    // as its label value and that String contained a comma. But that doesn't seem like it makes
    // a ton of sense.  Anyway, that's why I made the labelSplitFunction an input. But I don't
    // know if it helps much.
    Family labelFamily = registry.family(LABEL);
    Map<String, Double> results = labelSplitFunction.apply(value);
    Set<FeatureValue> values = new HashSet<>();
    for (Map.Entry<String, Double> entry : results.entrySet()) {
      values.add(new SimpleFeatureValue(labelFamily.feature(entry.getKey()),
                                             entry.getValue()));
    }
    return values;
  }

  // TODO (Brad): This is really gross. Let's change the way we handle labels to something that
  // doesn't involve such tortuous naming conventions.
  public static Map<String, Double> genericLabelSplitFunction(String label) {
    String[] labels = label.split(",");
    Map<String, Double> results = new HashMap<>();
    if (labels.length == 1) {
      // This shouldn't really happen. If it's not comma-delimited, it should be a double value.
      // But this should work for String labels just in case.
      results.put(label, 1.0);
      return results;
    }
    for (String s : labels) {
      String[] labelTokens = s.split(":");
      checkArgument(labelTokens.length == 2,
                    "MultiClass LABEL \"%s\" not in format [label1]:[weight1],...!", label);
      results.put(labelTokens[0], Double.valueOf(labelTokens[1]));
    }
    return results;
  }

  private static Feature parseFeature(String name, Object value, FeatureRegistry registry) {
    int pos = name.indexOf(FAMILY_SEPARATOR);
    checkArgument(pos > 0,
                  "Column name %s is invalid. It must either be %s or start with a family "
                  + "name followed by %s and a feature name.", name, LABEL, NAME_SEPARATOR);
    String familyName = name.substring(0, pos);
    String featureName = name.substring(pos + 1);
    if (value instanceof String) {
      if (featureName.equals(RAW)) {
        featureName = (String) value;
      } else {
        featureName = featureName + NAME_SEPARATOR + value;
      }
    } else if (value instanceof Boolean) {
      featureName = featureName + NAME_SEPARATOR + ((boolean)value ? TRUE_FEATURE : FALSE_FEATURE);
    }
    return registry.feature(familyName, featureName);
  }

  private static NamingConventionResult createNamingConventionResult(Feature feature,
                                                                     Object value) {
    if (value instanceof Number) {
      double val = ((Number) value).doubleValue();
      return NamingConventionResult.builder()
          .doubleFeatures(ImmutableSet.of(new SimpleFeatureValue(feature, val)))
          .build();
    } else {
      return NamingConventionResult.builder()
          .stringFeatures(ImmutableSet.of(feature))
          .build();
    }
  }


}
