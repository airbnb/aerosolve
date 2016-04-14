package com.airbnb.aerosolve.core.features;

import it.unimi.dsi.fastutil.objects.AbstractObject2DoubleMap;
import it.unimi.dsi.fastutil.objects.AbstractObjectIterator;
import it.unimi.dsi.fastutil.objects.AbstractObjectSet;
import it.unimi.dsi.fastutil.objects.AbstractReference2DoubleMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import it.unimi.dsi.fastutil.objects.ObjectSet;
import java.io.Serializable;
import java.util.Iterator;
import java.util.NoSuchElementException;
import lombok.Getter;
import lombok.Synchronized;
import lombok.experimental.Accessors;

/**
 *
 */
public class SparseVector extends Object2DoubleOpenHashMap<Feature> implements FamilyVector,
                                                                               Serializable {

  @Getter
  @Accessors(fluent = true)
  private final Family family;
  @Getter
  @Accessors(fluent = true)
  private final FeatureRegistry registry;

  public SparseVector(Family family, FeatureRegistry registry) {
    super(family.allocationSize());
    this.family = family;
    this.registry = registry;
  }
  @Override
  public Iterator<FeatureValue> iterator() {
    // TODO (Brad): We use fast for speed and memory usage when using foreach.  But it's a bit
    // dangerous because it mutates the instance returned from the iterator.
    return new SparseVectorIterator(this.object2DoubleEntrySet().iterator(), false);
  }

  @Override
  public Iterator<FeatureValue> fastIterator() {
    return new SparseVectorIterator(this.object2DoubleEntrySet().fastIterator(), true);
  }

  private class SparseVectorIterator implements Iterator<FeatureValue> {

    private final boolean goFast;
    private final ObjectIterator<Entry<Feature>> iterator;
    private final SimpleFeatureValueEntry value;

    public SparseVectorIterator(ObjectIterator<Entry<Feature>> iterator,
                                boolean goFast) {
      this.iterator = iterator;
      this.goFast = goFast;
      this.value = new SimpleFeatureValueEntry(null, 0.0d);
    }

    @Override
    public boolean hasNext() {
      return iterator.hasNext();
    }

    @Override
    public FeatureValueEntry next() {
      Entry<Feature> entry = iterator.next();
      if (goFast) {
        value.feature(entry.getKey());
        value.value(entry.getDoubleValue());
        return value;
      } else {
        return new SimpleFeatureValueEntry(entry.getKey(), entry.getDoubleValue());
      }
    }
  }
}
