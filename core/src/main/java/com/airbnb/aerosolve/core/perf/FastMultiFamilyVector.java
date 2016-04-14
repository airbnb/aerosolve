package com.airbnb.aerosolve.core.perf;

import com.airbnb.aerosolve.core.ThriftFeatureVector;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Doubles;
import it.unimi.dsi.fastutil.objects.AbstractObjectIterator;
import it.unimi.dsi.fastutil.objects.AbstractObjectSet;
import it.unimi.dsi.fastutil.objects.AbstractReference2DoubleMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;
import it.unimi.dsi.fastutil.objects.ObjectSet;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;

/**
 *
 */
public class FastMultiFamilyVector extends AbstractReference2DoubleMap<Feature> implements MultiFamilyVector {

  private final FeatureRegistry registry;
  private FamilyVector[] families;
  private int totalSize;

  public FastMultiFamilyVector(FeatureRegistry registry) { //featureRegistry.familyCount()
    this.families = new FamilyVector[registry.familyCapacity()];
    this.totalSize = 0;
    this.registry = registry;
  }

  public FastMultiFamilyVector(FastMultiFamilyVector other) {
    this.families = Arrays.copyOf(other.families, other.families.length);
    this.totalSize = other.totalSize;
    this.registry = other.registry;
  }

  public FastMultiFamilyVector(ThriftFeatureVector tmp, FeatureRegistry registry) {
    this(registry);
    if (tmp.getStringFeatures() != null) {
      for (Map.Entry<String, Set<String>> entry : tmp.getStringFeatures().entrySet()) {
        Family family = registry.family(entry.getKey());
        for (String featureName : entry.getValue()) {
          putString(family.feature(featureName));
        }
      }
    }
    if (tmp.getDenseFeatures() != null) {
      for (Map.Entry<String, List<Double>> entry : tmp.getDenseFeatures().entrySet()) {
        Family family = registry.family(entry.getKey());
        putDense(family, Doubles.toArray(entry.getValue()));
      }
    }
    if (tmp.getFloatFeatures() != null) {
      for (Map.Entry<String, Map<String, Double>> entry : tmp.getFloatFeatures().entrySet()) {
        Family family = registry.family(entry.getKey());
        for (Map.Entry<String, Double> value : entry.getValue().entrySet()) {
          put(family.feature(value.getKey()), (double) value.getValue());
        }
      }
    }
  }

  @Override
  public void applyContext(MultiFamilyVector context) {
    // TODO (Brad): For now, we copy values in.  Ideally, we would instead have this back the vector
    // and incur the hit on lookup to save memory and time.
    for (FamilyVector vector : context.families()) {
      if (vector instanceof DenseVector) {
        if (!contains(vector.family())) {
          putDense(vector.family(), ((DenseVector)vector).getValues());
        }
      } else {
        for (FeatureValue value : vector) {
          if (!containsKey(value.feature())) {
            put(value.feature(), value.getDoubleValue());
          }
        }
      }
    }
  }

  @Override
  public Set<? extends FamilyVector> families() {
    Set<FamilyVector> familySet = new HashSet<>();
    for (FamilyVector vector : families) {
      if (vector != null) {
        familySet.add(vector);
      }
    }
    return familySet;
  }

  @Override
  public void putString(Feature feature) {
    put(feature, 1.0d);
  }

  @Override
  public FamilyVector putDense(Family family, double[] values) {
    DenseVector fVec = (DenseVector) family(family, true);
    fVec.setValues(values);
    return fVec;
  }

  @Override
  public double put(Feature feature, double value) {
    FamilyVector familyVector = family(feature.family(), false);
    return familyVector.put(feature, value);
  }

  private FamilyVector family(Family family, boolean isDense) {
    if (families.length <= family.index()) {
      int newNumFamilies = families.length * 2;
      families = Arrays.copyOf(families, newNumFamilies);
    }
    FamilyVector fVec = families[family.index()];
    if (fVec == null) {
      if (isDense) {
        fVec = new DenseVector(family);
      } else {
        // TODO (Brad): Maybe have a small one and a large one?
        fVec = new SparseVector(family);
      }
      families[family.index()] = fVec;
    }
    return fVec;
  }

  @Override
  public FeatureRegistry registry() {
    return registry;
  }

  @Override
  public double removeDouble(Object key) {
    if (!(key instanceof Feature)) {
      return 0.0d;
    }
    Feature feature = (Feature) key;
    FamilyVector fVec = families[feature.family().index()];
    if (fVec != null) {
      double result = fVec.removeDouble(feature);
      if (fVec.size() == 0) {
        remove(feature.family());
      }
      return result;
    }
    return 0.0d;
  }

  @Override
  public FamilyVector remove(Family family) {
    if (family.index() >= families.length) {
      return null;
    }

    FamilyVector fVec = families[family.index()];
    families[family.index()] = null;
    return fVec;
  }

  @Override
  public FamilyVector get(Family family) {
    if (family.index() >= families.length) {
      return null;
    }
    return families[family.index()];
  }

  @Override
  public boolean containsKey(Object k) {
    if (!(k instanceof Feature)) {
      return false;
    }
    FamilyVector vec = get(((Feature) k).family());
    return vec != null && vec.containsKey(k);
  }

  @Override
  public boolean contains(Family family) {
    return family.index() < families.length && families[family.index()] != null;
  }

  @Override
  public int size() {
    int size = 0;
    for (FamilyVector fVec : families) {
      if (fVec != null) {
        size += fVec.size();
      }
    }
    return size;
  }

  @Override
  public double getDouble(Object key) {
    if (key instanceof Feature) {
      Feature feature = (Feature) key;
      if (feature.family().index() < families.length) {
        FamilyVector fVec = families[feature.family().index()];
        if (fVec != null) {
          return fVec.getDouble(feature);
        }
      }
    }
    return 0.0d;
  }

  @Override
  public Iterator<FeatureValue> iterator() {
    return (Iterator) new FastMultiFamilyVectorIterator(false);
  }

  @Override
  public Iterator<FeatureValue> fastIterator() {
    return (Iterator) new FastMultiFamilyVectorIterator(true);
  }

  @Override
  public ObjectSet<Entry<Feature>> reference2DoubleEntrySet() {
    return new FastFeatureVectorEntrySet();
  }

  public class FastFeatureVectorEntrySet extends AbstractObjectSet<Entry<Feature>> implements
      FastEntrySet<Feature> {

    @Override
    public ObjectIterator<Entry<Feature>> iterator() {
      return (ObjectIterator) new FastMultiFamilyVectorIterator(false);
    }

    @Override
    public int size() {
      return FastMultiFamilyVector.this.size();
    }

    @Override
    public ObjectIterator<Entry<Feature>> fastIterator() {
      return (ObjectIterator) new FastMultiFamilyVectorIterator(true);
    }
  }

  private class FastMultiFamilyVectorIterator extends
                                              AbstractObjectIterator<FeatureValueEntry> {
    private int index = -1;
    private Iterator<FeatureValueEntry> iterator;
    private final boolean useFast;
    private Iterator<FeatureValueEntry> nextIterator;
    private int nextIndex = -1;

    public FastMultiFamilyVectorIterator(boolean useFast) {
      this.useFast = useFast;
    }

    private void nextIterator() {
      nextIndex = index;
      while ((nextIterator == null || !nextIterator.hasNext()) && nextIndex < families.length - 1) {
        nextIndex++;
        if (families[nextIndex] != null) {
          nextIterator = useFast
                         ? (Iterator) families[nextIndex].fastIterator()
                         : (Iterator) families[nextIndex].iterator();
        }
      }
    }

    @Override
    public boolean hasNext() {
      if (iterator != null && iterator.hasNext()) {
        return true;
      }
      if (nextIterator != null) {
        return nextIterator.hasNext();
      }
      nextIterator();
      return nextIterator != null && nextIterator.hasNext();
    }

    @Override
    public FeatureValueEntry next() {
      if (iterator != null && iterator.hasNext()) {
        return iterator.next();
      }
      if (nextIterator == null) {
        nextIterator();
      }
      iterator = nextIterator;
      index = nextIndex;
      nextIterator = null;
      if (iterator == null) {
        throw new NoSuchElementException();
      }
      return iterator.next();
    }
  }
}
