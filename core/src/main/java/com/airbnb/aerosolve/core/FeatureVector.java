package com.airbnb.aerosolve.core;

import com.airbnb.aerosolve.core.perf.Feature;
import com.airbnb.aerosolve.core.perf.FeatureValue;
import com.airbnb.aerosolve.core.perf.SimpleFeatureValueEntry;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import it.unimi.dsi.fastutil.objects.Reference2DoubleMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Consumer;

/**
 * When iterating a FeatureVector it may not always return a new copy of each value.  If you want
 * to save the values returned by the iterator, use the entry set instead.
 */
public interface FeatureVector extends Reference2DoubleMap<Feature>, Iterable<FeatureValue> {
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

  default Iterable<FeatureValue> withDropout(double dropout) {
    return Iterables.filter(this, f -> ThreadLocalRandom.current().nextDouble() < dropout);
  }

  default FeatureValue getFeatureValue(Feature feature) {
    return new SimpleFeatureValueEntry(feature, getDouble(feature));
  }
}
