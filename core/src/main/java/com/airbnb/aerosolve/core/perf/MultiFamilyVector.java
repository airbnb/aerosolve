package com.airbnb.aerosolve.core.perf;

import com.airbnb.aerosolve.core.FeatureVector;
import java.util.Set;

/**
 *
 */
public interface MultiFamilyVector extends FeatureVector {

  FamilyVector putDense(Family family, double[] values);

  FamilyVector remove(Family family);

  FamilyVector get(Family family);

  boolean contains(Family family);

  FeatureRegistry registry();

  void applyContext(MultiFamilyVector context);

  Set<? extends FamilyVector> families();

  default int numFamilies() {
    return families().size();
  }
}
