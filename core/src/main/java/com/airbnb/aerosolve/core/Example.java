package com.airbnb.aerosolve.core;

import com.airbnb.aerosolve.core.models.AbstractModel;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.Transformer;
import java.util.Iterator;

/**
 *
 */
public interface Example extends Iterable<MultiFamilyVector> {
  MultiFamilyVector context();

  MultiFamilyVector createVector();

  MultiFamilyVector addToExample(MultiFamilyVector vector);

  Example transform(Transformer transformer,
                 AbstractModel model);

  default Example transform(Transformer transformer) {
    return transform(transformer, null);
  }

  /**
   * Returns the only MultiFamilyVector in this Example.
   *
   * If the Example contains nothing or more than one thing, this will throw an
   * IllegalStateException.
   *
   * (Brad): Lots of code assumes the Example has only one item.  Ideally, we should remove that
   * assumption and then this method.  This method helps us find code paths making that assumption.
   */
  default MultiFamilyVector only() {
    Iterator<MultiFamilyVector> iterator = iterator();
    if (!iterator.hasNext()) {
      throw new IllegalStateException("Called only() on an Example which contains nothing");
    }
    MultiFamilyVector result = iterator.next();
    if (iterator.hasNext()) {
      throw new IllegalStateException("Called only() on an Example containing more than one vector.");
    }
    return result;
  }
}
