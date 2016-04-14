package com.airbnb.aerosolve.core;

import com.airbnb.aerosolve.core.models.AbstractModel;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.Transform;
import com.airbnb.aerosolve.core.transforms.Transformer;

/**
 *
 */
public interface Example extends Iterable<MultiFamilyVector> {
  MultiFamilyVector context();

  MultiFamilyVector createVector();

  MultiFamilyVector addToExample(MultiFamilyVector vector);

  void transform(Transformer transformer,
                 AbstractModel model);

  default void transform(Transformer transformer) {
    transform(transformer, null);
  }
}
