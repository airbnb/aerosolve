package com.airbnb.aerosolve.core.perf;

import com.google.common.primitives.Ints;
import it.unimi.dsi.fastutil.objects.Reference2ObjectMap;
import it.unimi.dsi.fastutil.objects.Reference2ObjectOpenHashMap;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import lombok.Value;

/**
 *
 *
 *
 * There's some awkward code in here that might seem to be better handled by inheritance.
 * The trouble is that we use interning of Strings to Families and then use reference equality for
 * speed.  Sometimes we have to create families with types that are not resolved yet (when
 * deserializing models.) If we used inheritance, we might be tempted to return a new instance
 * with the correct subtype when the type is resolved.  But this would break reference equality.
 *
 * Instead we mutate the internal state and behavior of the object using the type parameter. That
 * serves as a weak form of polymorphism.  We could also make this object a proxy for an internal
 * correctly typed object.  But since there are only two interesting possible subclasses right now
 * it  seemed simpler to just have all the logic in the same class.
 */
public class BasicFamily implements Family {

  private Map<String, Feature> featuresByName;
  private final String name;
  private final int index;
  private final AtomicInteger featureCount;
  private Feature[] featuresByIndex;
  private boolean isDense = false;
  private Reference2ObjectMap<Feature, Reference2ObjectMap<Feature, FeatureJoin[]>> crosses;

  BasicFamily(String name, int index) {
    this.name = name;
    this.index = index;
    this.featureCount = new AtomicInteger(0);
    this.crosses = new Reference2ObjectOpenHashMap<>();
  }

  @Override
  public int index() {
    return index;
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public void makeDense() {
    if (featuresByName != null && !featuresByName.isEmpty()) {
      throw new IllegalStateException("Tried to make a family dense but it already has features"
                                      + " defined by name. Some code probably thinks it's sparse.");
    }
    isDense = true;
  }

  @Override
  public Feature feature(String featureName) {
    if (isDense) {
      // Note that it's not recommended to use this method if the type is DENSE.
      // Just call feature(int). It will be faster.
      Integer index = Ints.tryParse(featureName);
      if (index == null) {
        throw new IllegalArgumentException(String.format(
            "Could not parse %s to a valid integer for lookup in a dense family: %s. Dense families "
            + "do not have names for each feature.", featureName, name()));
      }
      return feature(index);
    }
    if (featuresByName == null) {
      featuresByName = new ConcurrentHashMap<>(allocationSize());
    }
    Feature feature = featuresByName.computeIfAbsent(
        featureName,
        innerName -> new Feature(this, innerName, featureCount.getAndIncrement())
    );
    resizeFeaturesByIndex(featureCount.get());
    if (featuresByIndex[feature.index()] == null) {
      featuresByIndex[feature.index()] = feature;
    }
    return feature;
  }

  @Override
  public Feature feature(int index) {
    if (featuresByIndex == null || index >= featuresByIndex.length) {
      if (isDense) {
        resizeFeaturesByIndex(index);
      } else {
        return null;
      }
    }
    if (isDense && featuresByIndex[index] == null) {
      featuresByIndex[index] = new Feature(this, String.valueOf(index), index);
    }
    return featuresByIndex[index];
  }

  private void resizeFeaturesByIndex(int index) {
    if (featuresByIndex == null) {
      featuresByIndex = new Feature[Family.allocationSize(index + 1)];
      return;
    }
    if (index < featuresByIndex.length) {
      return;
    }
    // Need to resize.
    int length = featuresByIndex.length;
    while (index >= length) {
      length = length * 2;
    }
    // TODO (Brad): Synchronize.
    featuresByIndex = Arrays.copyOf(featuresByIndex, length);
  }

  @Override
  public Feature cross(Feature left, Feature right, String separator) {
    Reference2ObjectMap<Feature, FeatureJoin[]> rightMap = crosses.get(left);
    if (rightMap == null) {
      rightMap = new Reference2ObjectOpenHashMap<>();
      crosses.put(left, rightMap);
    }
    FeatureJoin[] joinArr = rightMap.get(right);
    if (joinArr == null) {
      joinArr = new FeatureJoin[2];
      rightMap.put(right, joinArr);
    }
    int i;
    for (i = 0; i < joinArr.length; i++) {
      FeatureJoin join = joinArr[i];
      if (join == null) {
        break;
      }
      if (join.getSeparator().equals(separator)) {
        return join.getFeature();
      }
    }
    Feature feature = feature(left.name() + separator + right.name());
    FeatureJoin newJoin = new FeatureJoin(feature, separator);
    if (i >= joinArr.length) {
      joinArr = Arrays.copyOf(joinArr, joinArr.length * 2);
      rightMap.put(right, joinArr);
    }
    joinArr[i] = newJoin;
    return feature;
  }

  public int size() {
    return featureCount.get();
  }

  @Override
  public int hashCode() {
    return getHashCode();
  }

  @Override
  public boolean equals(Object obj) {
    return isEqual(obj);
  }

  @Override
  public String toString() {
    return name;
  }

  @Value
  private static class FeatureJoin {
    private final Feature feature;
    private final String separator;
  }
}
