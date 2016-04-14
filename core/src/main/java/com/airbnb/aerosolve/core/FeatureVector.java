package com.airbnb.aerosolve.core;

import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.SimpleFeatureValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Consumer;

/**
 * When iterating a FeatureVector it may not always return a new copy of each value.  If you want
 * to save the values returned by the iterator, use the entry set instead.
 */
public interface FeatureVector extends Object2DoubleMap<Feature>, Iterable<FeatureValue> {
  FeatureRegistry registry();

  default void putString(Feature feature) {
    put(feature, 1.0d);
  }

  default Iterator<FeatureValue> fastIterator() {
    return iterator();
  }

  /**
   * Use this if you intend to store the values. Don't use foreach.
   */
  default Set<FeatureValue> featureValueEntrySet() {
    return Sets.newHashSet(iterator());
  }

  @Override
  default void forEach(Consumer<? super FeatureValue> action) {
    Iterator<FeatureValue> iter = fastIterator();
    while (iter.hasNext()) {
      action.accept(iter.next());
    }
  }

  default double get(String familyName, String featureName) {
    Feature feature = registry().feature(familyName, featureName);
    return getDouble(feature);
  }

  default boolean containsKey(String familyName, String featureName) {
    Feature feature = registry().feature(familyName, featureName);
    return containsKey(feature);
  }

  // TODO (Brad): This kind of breaks the abstraction. Do all Features have families?
  default void put(String familyName, String featureName, double value) {
    Feature feature = registry().feature(familyName, featureName);
    put(feature, value);
  }

  default void putString(String familyName, String featureName) {
    Feature feature = registry().feature(familyName, featureName);
    putString(feature);
  }

  default Iterable<FeatureValue> withDropout(double dropout) {
    return Iterables.filter(this, f -> ThreadLocalRandom.current().nextDouble() >= dropout);
  }

  default Iterable<FeatureValue> iterateMatching(List<Feature> features) {
    Preconditions.checkNotNull(features, "Cannot iterate all features when features is null");
    return () -> new FeatureSetIterator(this, features);
  }

  default double[] denseArray() {
    double[] result = new double[size()];
    int i = 0;
    for (FeatureValue value : this) {
      result[i] = value.value();
      i++;
    }
    return result;
  }

  class FeatureSetIterator implements Iterator<FeatureValue> {

    private final FeatureVector vector;
    private final List<Feature> features;
    private SimpleFeatureValue entry = SimpleFeatureValue.of(null, 0.0d);
    private int index = -1;
    private int nextIndex = 0;

    public FeatureSetIterator(FeatureVector vector, List<Feature> features) {
      this.vector = vector;
      this.features = features;
    }

    @Override
    public boolean hasNext() {
      if (index >= nextIndex) {
        nextIndex++;
        while(nextIndex < features.size() && !vector.containsKey(features.get(nextIndex))) {
          nextIndex++;
        }
      }
      return nextIndex < features.size();
    }

    @Override
    public FeatureValue next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      entry.feature(features.get(nextIndex));
      entry.value(vector.getDouble(entry.feature()));
      index = nextIndex;
      return entry;
    }
  }
}
