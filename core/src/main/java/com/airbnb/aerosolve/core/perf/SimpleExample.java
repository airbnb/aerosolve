package com.airbnb.aerosolve.core.perf;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ThriftExample;
import com.airbnb.aerosolve.core.ThriftFeatureVector;
import com.airbnb.aerosolve.core.models.AbstractModel;
import com.airbnb.aerosolve.core.transforms.Transform;
import com.airbnb.aerosolve.core.transforms.Transformer;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 *
 */
public class SimpleExample implements Example {

  private final MultiFamilyVector context;
  private final List<MultiFamilyVector> vectors;
  private final FeatureRegistry registry;

  public SimpleExample(FeatureRegistry registry) {
    this.context = new FastMultiFamilyVector(registry);
    this.vectors = new ArrayList<>();
    this.registry = registry;
  }

  public SimpleExample(ThriftExample example, FeatureRegistry registry) {
    this.registry = registry;
    if (example.getContext() != null) {
      this.context = new FastMultiFamilyVector(example.getContext(), registry);
    } else {
      this.context = new FastMultiFamilyVector(registry);
    }
    this.vectors = new ArrayList<>(example.getExampleSize());
    if (example.getExample() != null) {
      for (ThriftFeatureVector vector : example.getExample()) {
        vectors.add(new FastMultiFamilyVector(vector, registry));
      }
    }
  }

  @Override
  public MultiFamilyVector context() {
    return context;
  }

  @Override
  public MultiFamilyVector createVector() {
    return addToExample(new FastMultiFamilyVector(registry));
  }

  @Override
  public MultiFamilyVector addToExample(MultiFamilyVector vector) {
    vectors.add(vector);
    return vector;
  }

  @Override
  public Iterator<MultiFamilyVector> iterator() {
    return vectors.iterator();
  }

  @Override
  public void transform(Transformer transformer, AbstractModel model) {
    // TODO (Brad): Enable immutability by handling the return value.
    if (transformer.getContextTransform() != null) {
      transformer.getContextTransform().apply(context);
    }
    for (MultiFamilyVector vector : vectors) {
      if (transformer.getItemTransform() != null) {
        transformer.getItemTransform().apply(vector);
      }
      if (transformer.getCombinedTransform() != null) {
        // TODO (Brad): REVISIT THIS BEFORE MERGING!!
        // Ideally we wouldn't copy the context into each vector but instead have it back the
        // vector as a fallback. This would avoid the costs of copying and the memory usage.
        // But it might be slower.  Need to test.
        // This changes the semantics of context. Previously an entire context family would
        // overwrite an item family.  This is a merge that favors features
        // in the item.  I think this is a better semantics as it's more useful in the cases I've
        // seen but I'm not sure what other cases people have encountered.
        vector.applyContext(context);
        transformer.getCombinedTransform().apply(vector);
      }
    }
  }

}
