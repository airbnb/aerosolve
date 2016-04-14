package com.airbnb.aerosolve.core.features;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.objects.Reference2ObjectMap;
import it.unimi.dsi.fastutil.objects.Reference2ObjectOpenHashMap;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import lombok.Getter;
import lombok.Synchronized;
import lombok.Value;
import lombok.experimental.Accessors;

/**
 *
 */
@Accessors(fluent = true, chain = true)
public class BasicFamily implements Family, Serializable {

  private final int hashCode;
  private Map<String, Feature> featuresByName;
  @Getter
  private final String name;
  @Getter
  private final int index;
  private final AtomicInteger featureCount;
  private Feature[] featuresByIndex;
  @Getter
  private boolean isDense = false;
  private Map<Feature, Map<Feature, FeatureJoin[]>> crosses;

  BasicFamily(String name, int index) {
    Preconditions.checkNotNull(name, "All Families must have a name");
    this.name = name;
    this.index = index;
    this.featureCount = new AtomicInteger(0);
    this.crosses = new Object2ObjectOpenHashMap<>();
    this.hashCode = name.hashCode();
  }

  @Override
  public void markDense() {
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
    if (featuresByIndex == null || feature.index() >= featuresByIndex.length) {
      resizeFeaturesByIndex(feature.index());
    }
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

  @Synchronized
  private void resizeFeaturesByIndex(int index) {
    if (featuresByIndex == null) {
      featuresByIndex = new Feature[Family.allocationSize(index + 1)];
      return;
    }
    // We check outside and inside this method because it's synchronized and this can change
    // between when we intend to enter and when we actually enter.
    if (index < featuresByIndex.length) {
      return;
    }
    // Need to resize.
    int length = featuresByIndex.length;
    while (index >= length) {
      length = length * 2;
    }
    featuresByIndex = Arrays.copyOf(featuresByIndex, length);
  }

  @Override
  public Feature cross(Feature left, Feature right, String separator) {
    Map<Feature, FeatureJoin[]> rightMap = crosses.get(left);
    if (rightMap == null) {
      rightMap = new Object2ObjectOpenHashMap<>();
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
      if (join.separator().equals(separator)) {
        return join.feature();
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
    return hashCode;
  }

  @Override
  public boolean equals(Object obj) {
    if (obj == this) {
      return true;
    }
    if (!(obj instanceof Family)) {
      return false;
    }
    return name.equals(((Family) obj).name());
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
