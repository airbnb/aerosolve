package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.BasicMultiFamilyVector;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;

public class TransformTestingHelper {
  public static MultiFamilyVector makeEmptyVector() {
    FeatureRegistry registry = new FeatureRegistry();
    return makeEmptyVector(registry);
  }

  public static MultiFamilyVector makeEmptyVector(FeatureRegistry registry) {
    return new BasicMultiFamilyVector(registry);
  }

  public static MultiFamilyVector makeSimpleVector(FeatureRegistry registry) {
    return builder(registry)
        .simpleStrings()
        .location()
        .build();
  }

  public static MultiFamilyVector makeFoobarVector(FeatureRegistry registry) {
    return builder(registry)
        .simpleStrings()
        .location()
        .foobar()
        .build();
  }

  public static VectorBuilder builder(FeatureRegistry registry) {
    return new VectorBuilder(registry);
  }

  public static VectorBuilder builder(FeatureRegistry registry, MultiFamilyVector vector) {
    return new VectorBuilder(registry, vector);
  }

  // Not actually a real builder. But calling things twice should be idempotent so . . .
  public static class VectorBuilder {
    private final FeatureRegistry registry;
    private final MultiFamilyVector vector;

    public VectorBuilder(FeatureRegistry registry) {
      this(registry, new BasicMultiFamilyVector(registry));
    }

    public VectorBuilder(FeatureRegistry registry, MultiFamilyVector vector) {
      this.registry = registry;
      this.vector = vector;
    }

    public MultiFamilyVector build() {
      return vector;
    }

    public VectorBuilder sparse(String family, String name, double value) {
      vector.put(registry.feature(family, name), value);
      return this;
    }

    public VectorBuilder string(String family, String name) {
      vector.putString(registry.feature(family, name));
      return this;
    }

    public VectorBuilder dense(String family, double[] values) {
      vector.putDense(registry.family(family), values);
      return this;
    }

    public VectorBuilder simpleStrings() {
      return this
          .string("strFeature1", "aaa")
          .string("strFeature1", "bbb");
    }

    public VectorBuilder location() {
      return this
          .sparse("loc", "lat", 37.7)
          .sparse("loc", "long", 40.0);
    }

    public VectorBuilder foobar() {
      return this
          .sparse("loc", "z", -20.0)
          .sparse("F", "foo", 1.5)
          .sparse("bar", "bar_fv", 1.0);
    }

    public VectorBuilder complexLocation() {
      return this
          .sparse("loc", "a", 0.0)
          .sparse("loc", "b", 0.13)
          .sparse("loc", "c", 1.23)
          .sparse("loc", "d", 5.0)
          .sparse("loc", "e", 17.5)
          .sparse("loc", "f", 99.98)
          .sparse("loc", "g", 365.0)
          .sparse("loc", "h", 65537.0)
          .sparse("loc", "i", -1.0)
          .sparse("loc", "j", -23.0);
    }
  }
}
