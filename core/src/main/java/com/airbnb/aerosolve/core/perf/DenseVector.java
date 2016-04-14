package com.airbnb.aerosolve.core.perf;

import it.unimi.dsi.fastutil.objects.AbstractObjectIterator;
import it.unimi.dsi.fastutil.objects.AbstractObjectSet;
import it.unimi.dsi.fastutil.objects.AbstractReference2DoubleMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import it.unimi.dsi.fastutil.objects.ObjectSet;
import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 *
 */
public class DenseVector extends AbstractReference2DoubleMap<Feature> implements FamilyVector {

  private final Family family;
  private double[] values;

  public DenseVector(Family family) {
    family.makeDense();
    this.family = family;
  }

  @Override
  public Family family() {
    return family;
  }

  public void setValues(double[] values) {
    this.values = values;
  }

  public double[] getValues() {
    return values;
  }

  @Override
  public ObjectSet<Entry<Feature>> reference2DoubleEntrySet() {
    return new DenseVectorEntrySet();
  }

  @Override
  public int size() {
    return values.length;
  }

  @Override
  public double getDouble(Object key) {
    if (key instanceof Feature) {
      int featureIndex = ((Feature) key).index();
      if (featureIndex < values.length) {
        return values[featureIndex];
      }
    }
    return 0.0d;
  }

  @Override
  public boolean containsKey(Object k) {
    return k instanceof Feature && ((Feature) k).index() < values.length;
  }

  @Override
  public double put(Feature key, double value) {
    throw new UnsupportedOperationException("Don't put to a dense vector. Call setValues().");
  }

  @Override
  public double removeDouble(Object key) {
    throw new UnsupportedOperationException("Cannot remove values from a DenseFamilyVector");
  }

  @Override
  public Iterator<FeatureValue> iterator() {
    return fastIterator();
  }

  @Override
  public Iterator<FeatureValue> fastIterator() {
    return (Iterator) new FastDenseVectorIterator();
  }

  private class DenseVectorEntrySet extends AbstractObjectSet<Entry<Feature>> implements
    FastEntrySet<Feature> {
    @Override
    public ObjectIterator<Entry<Feature>> iterator() {
      return fastIterator();
    }

    @Override
    public int size() {
      return values.length;
    }

    @Override
    public ObjectIterator<Entry<Feature>> fastIterator() {
      return (ObjectIterator) new FastDenseVectorIterator();
    }
  }

  private class FastDenseVectorIterator extends AbstractObjectIterator<FeatureValueEntry> {

    private int index = -1;
    private FastEntry entry = new FastEntry();

    @Override
    public boolean hasNext() {
      return index < values.length - 1;
    }

    @Override
    public FeatureValueEntry next() {
      if (hasNext()) {
        index++;
      } else {
        throw new NoSuchElementException();
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
        return family.feature(index);
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
    }
  }


}
