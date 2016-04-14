package com.airbnb.aerosolve.core.features;

import it.unimi.dsi.fastutil.objects.AbstractObject2DoubleMap;
import it.unimi.dsi.fastutil.objects.AbstractObjectIterator;
import it.unimi.dsi.fastutil.objects.AbstractObjectSet;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import it.unimi.dsi.fastutil.objects.ObjectSet;
import java.io.Serializable;
import java.util.Iterator;
import java.util.NoSuchElementException;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.Accessors;

/**
 *
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
public class DenseVector extends AbstractObject2DoubleMap<Feature>
    implements FamilyVector, Serializable {

  private final Family family;
  private final FeatureRegistry registry;
  private double[] denseArray;

  public DenseVector(Family family, FeatureRegistry registry) {
    family.markDense();
    this.family = family;
    this.registry = registry;
  }

  @Override
  public ObjectSet<Entry<Feature>> object2DoubleEntrySet() {
    return new DenseVectorEntrySet();
  }

  @Override
  public int size() {
    return denseArray.length;
  }

  @Override
  public double getDouble(Object key) {
    if (key instanceof Feature) {
      int featureIndex = ((Feature) key).index();
      if (featureIndex < denseArray.length) {
        return denseArray[featureIndex];
      }
    }
    return 0.0d;
  }

  @Override
  public boolean containsKey(Object k) {
    return k instanceof Feature && ((Feature) k).index() < denseArray.length;
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
      return denseArray.length;
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
      return index < denseArray.length - 1;
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
      @Deprecated
      public Double getValue() {
        return denseArray[index];
      }

      @Override
      public double setValue(double value) {
        throw new UnsupportedOperationException();
      }

      @Override
      public double value() {
        return denseArray[index];
      }

      @Override
      public double getDoubleValue() {
        return value();
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
      public void feature(Feature feature) {
        throw new UnsupportedOperationException();
      }

      @Override
      public String toString() {
        return feature() + "::" + value();
      }
    }
  }


}
