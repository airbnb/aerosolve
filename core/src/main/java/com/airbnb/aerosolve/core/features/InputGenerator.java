package com.airbnb.aerosolve.core.features;

/*
  use Float.MIN_VALUE as NULL for the float feature.
 */
// TODO (Brad): I'm not sure this class belongs in core.  It's very specific to one way we generate
// our feature vectors from external data.  It seems like if we just tracked the String[] per class
// through some external mechanism, we could just call MultiFamilyVector.putAll(String[], Object[])
// every time we got a new Object[] or double[].
// For now, it's easy to make it work but I'm going to discuss with Julian before merging.
public class InputGenerator {
  private final InputSchema mapping;
  private Object[] values;

  public InputGenerator(InputSchema mapping) {
    this.mapping = mapping;
    values = new Object[mapping.getNames().length];
  }

  public void add(double[] features, Object c) {
    InputSchema.Entry e = mapping.getMapping().get(c);
    assert(e.length == features.length);
    // can't do System.arraycopy(features, 0, values, e.start, e.length);
    // due to Float.MIN_VALUE means NULL
    for (int i = 0; i < e.length; i++) {
      if (features[i] != Float.MIN_VALUE) {
        values[i + e.start] = features[i];
      }
    }
  }

  public void add(Object[] features, Object c) {
    InputSchema.Entry e = mapping.getMapping().get(c);
    assert(e.length == features.length);
    System.arraycopy(features, 0, values, e.start, e.length);
  }

  /**
   * Load the schema and values in this generator into a MultiFamilyVector.
   */
  public MultiFamilyVector load(MultiFamilyVector vector) {
    return vector.putAll(mapping.getNames(), values);
  }
}
