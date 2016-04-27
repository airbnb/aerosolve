package com.airbnb.aerosolve.core.features;

/*
  use Float.MIN_VALUE as NULL for the float feature.
 */
public class FeatureGen {
  private final FeatureMapping mapping;
  private Object[] values;

  public FeatureGen(FeatureMapping mapping) {
    this.mapping = mapping;
    values = new Object[mapping.getNames().length];
  }

  public void add(float[] features, Object c) {
    FeatureMapping.Entry e = mapping.getMapping().get(c);
    assert(e.length == features.length);
    // can't do System.arraycopy(features, 0, values, e.start, e.length);
    // due to Float.MIN_VALUE means NULL
    for (int i = 0; i < e.length; i++) {
      if (features[i] != Float.MIN_VALUE) {
        values[i + e.start] = new Double(features[i]);
      }
    }
  }

  public void add(Object[] features, Object c) {
    FeatureMapping.Entry e = mapping.getMapping().get(c);
    assert(e.length == features.length);
    System.arraycopy(features, 0, values, e.start, e.length);
  }

  public Features gen() {
    return Features.builder().
        names(mapping.getNames()).
        values(values).
        build();
  }
}
