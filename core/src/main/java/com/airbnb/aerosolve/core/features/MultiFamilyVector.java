package com.airbnb.aerosolve.core.features;

import com.airbnb.aerosolve.core.FeatureVector;
import java.util.Map;
import java.util.Set;

import javax.validation.constraints.NotNull;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

/**
 *
 */
public interface MultiFamilyVector extends FeatureVector {

  FamilyVector putDense(Family family, double[] values);

  FamilyVector remove(Family family);

  FamilyVector get(Family family);

  boolean contains(Family family);

  void applyContext(MultiFamilyVector context);

  Set<? extends FamilyVector> families();

  default MultiFamilyVector merge(MultiFamilyVector vector) {
    for (FamilyVector familyVector : vector.families()) {
      // Assuming we want to keep any existing values.
      if (familyVector.family().isDense()) {
        putDense(familyVector.family(), familyVector.denseArray());
      } else {
        for (FeatureValue value : familyVector) {
          put(value.feature(), value.value());
        }
      }
    }
    return this;
  }

  default FamilyVector putDense(String familyName, double[] values) {
    Family family = registry().family(familyName);
    return putDense(family, values);
  }

  default FamilyVector remove(String familyName) {
    Family family = registry().family(familyName);
    return remove(family);
  }

  default FamilyVector get(String familyName) {
    Family family = registry().family(familyName);
    return get(family);
  }

  default boolean contains(String familyName) {
    Family family = registry().family(familyName);
    return contains(family);
  }

  default MultiFamilyVector putAllObjects(Map<String, Object> features,
                                          NamingConvention namingConvention) {
    return putAll(features.keySet().toArray(new String[features.size()]),
                  features.values().toArray(),
                  namingConvention);

  }

  default MultiFamilyVector putAllObjects(Map<String, Object> features) {
    return putAllObjects(features, GenericNamingConvention.instance());
  }

  default MultiFamilyVector putAll(@NotNull String[] names, @NotNull Object[] values,
                                   NamingConvention namingConvention) {
    checkNotNull(names, "Names cannot be null when putting to a MultiFamilyVector");
    checkNotNull(values, "Values cannot be null when putting to a MultiFamilyVector");
    checkArgument(names.length == values.length,
                  "When putting arrays to a MultiFamilyVector, the names and values"
                  + " were of different sizes. Names size: %d Values size: %d",
                  names.length, values.length);
    for (int i = 0; i < names.length; i++) {
      NamingConvention.NamingConventionResult result =
          namingConvention.features(names[i], values[i], registry());
      if (result.getDenseFeatures() != null) {
        for (Map.Entry<Family, double[]> denseEntry : result.getDenseFeatures().entrySet()) {
          putDense(denseEntry.getKey(), denseEntry.getValue());
        }
      }
      if (result.getStringFeatures() != null) {
        for (Feature feature : result.getStringFeatures()) {
          putString(feature);
        }
      }
      if (result.getDoubleFeatures() != null) {
        for (FeatureValue value : result.getDoubleFeatures()) {
          put(value.feature(), value.value());
        }
      }
    }
    return this;
  }

  default MultiFamilyVector putAll(String[] names, Object[] values) {
    return putAll(names, values, GenericNamingConvention.instance());
  }

  default int numFamilies() {
    return families().size();
  }

  MultiFamilyVector withFamilyDropout(double dropout);
}
