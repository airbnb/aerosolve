package com.airbnb.aerosolve.core.perf;

import it.unimi.dsi.fastutil.objects.AbstractObjectIterator;
import it.unimi.dsi.fastutil.objects.AbstractObjectSet;
import it.unimi.dsi.fastutil.objects.AbstractReference2DoubleMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import it.unimi.dsi.fastutil.objects.ObjectSet;
import java.util.Iterator;
import java.util.NoSuchElementException;
import lombok.ToString;

/**
 *
 */
public class SparseVector extends AbstractReference2DoubleMap<Feature> implements FamilyVector {

  private final Family family;
  private Feature[] keys;
  private double[] values;
  private int size;

  public SparseVector(Family family) {
    this.family = family;
    int allocationSize = family.allocationSize();
    this.keys = new Feature[allocationSize];
    this.values = new double[allocationSize];
    this.size = 0;
  }

  @Override
  public Family family() {
    return family;
  }

  @Override
  public Iterator<FeatureValue> iterator() {

    // TODO (Brad): Profile and figure out how to not use this when calling foreach.
    return (Iterator) new SparseVectorIterator(false);
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public ObjectSet<Entry<Feature>> reference2DoubleEntrySet() {
    return new SparseVectorEntrySet();
  }

  @Override
  public double getDouble(Object key) {
    int index = -1;
    if (key instanceof Feature) {
      index = getIndex((Feature) key, keys);
    }
    return index == -1 ? 0.0d : values[index];
  }

  @Override
  public boolean containsKey(Object key) {
    return key instanceof Feature && getIndex((Feature) key, keys) != -1;
  }

  @Override
  public double put(Feature key, double value) {
    resizeIfNecessary();
    int index = probe(key, keys, false);
    if (keys[index] != key) {
      size++;
      keys[index] = key;
    }
    values[index] = value;
    return values[index];
  }

  private static int getIndex(Feature key, Feature[] keys) {
    return probe(key, keys, true);
  }

  private static int probe(Feature key, Feature[] keys, boolean match) {
    int index = key.index() % keys.length;
    int tries = 0;
    while(keys[index] != null && tries < keys.length) {
      if (keys[index] == key) {
        return index;
      }
      tries++;
      index = (index + (tries * tries)) % keys.length;
    }
    if (tries >= keys.length) {
      throw new IllegalStateException("Sparse Vector is full! There is a bug. We should have"
                                      + " resized so this couldn't happen.");
    }
    return match ? -1 : index;
  }

  private void resizeIfNecessary() {
    // TODO (Brad): Synchronize on resize.
    // Resize when we're over half full.
    if (size * 2 > keys.length) {
      int newSize = keys.length * 2;
      double[] newValues = new double[newSize];
      Feature[] newKeys = new Feature[newSize];
      for (int i = 0; i < keys.length; i++) {
        Feature feature = keys[i];
        if (feature != null) {
          int index = probe(feature, newKeys, false);
          newKeys[index] = feature;
          newValues[index] = values[i];
        }
      }
      values = newValues;
      keys = newKeys;
    }
  }

  @Override
  public double removeDouble(Object key) {
    if (key instanceof Feature) {
      Feature feature = (Feature) key;
      int index = getIndex(feature, keys);
      if (index != -1 && keys[index] == key) {
        keys[index] = null;
        values[index] = 0.0d;
        size--;
        return values[index];
      }
    }
    return 0.0d;
  }

  @Override
  public Iterator<FeatureValue> fastIterator() {
    return (Iterator) new SparseVectorIterator(true);
  }

  private class SparseVectorIterator extends AbstractObjectIterator<FeatureValueEntry> {
    private int index = -1;
    private boolean hasNext = false;
    private int nextIndex = -1;
    private FastEntry entry = new FastEntry();
    private final boolean goFast;

    public SparseVectorIterator(boolean goFast) {
      this.goFast = goFast;
    }

    @Override
    public boolean hasNext() {
      if (hasNext) {
        return true;
      }
      nextIndex();
      return hasNext;
    }

    private void nextIndex() {
      for (int i = index + 1; i < keys.length; i++) {
        if (keys[i] != null) {
          hasNext = true;
          nextIndex = i;
          return;
        }
      }
    }

    @Override
    public FeatureValueEntry next() {
      if (hasNext()) {
        index = nextIndex;
        hasNext = false;
      } else {
        throw new NoSuchElementException();
      }
      if (!goFast) {
        return entry.materialize();
      }
      return entry;
    }

    private class FastEntry implements FeatureValueEntry {

      @Override
      public Double getValue() {
        return values[index];
      }

      @Override
      public double setValue(double value) {
        throw new UnsupportedOperationException();
      }

      @Override
      public double getDoubleValue() {
        return values[index];
      }

      @Override
      public Feature getKey() {
        return keys[index];
      }

      @Override
      public Double setValue(Double value) {
        throw new UnsupportedOperationException();
      }

      @Override
      public Feature feature() {
        return getKey();
      }

      @Override
      public Feature setFeature(Feature feature) {
        throw new UnsupportedOperationException();
      }

      @Override
      public String toString() {
        return feature() + "::" + getDoubleValue();
      }

      public FeatureValueEntry materialize() {
        return new SimpleFeatureValueEntry(feature(), getDoubleValue());
      }
    }
  }

  private class SparseVectorEntrySet extends AbstractObjectSet<Entry<Feature>> implements
    FastEntrySet<Feature> {

    @Override
    public ObjectIterator<Entry<Feature>> iterator() {
      return (ObjectIterator) new SparseVectorIterator(false);
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public ObjectIterator<Entry<Feature>> fastIterator() {
      return (ObjectIterator) new SparseVectorIterator(true);
    }
  }
}
