package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.ModelRecord;

import java.util.List;

/**
 * This is a special case of a point where no contribution whatsoever is provided by this feature.
 * This is intended to mark feature as deleted for future deletion. This is needed so we avoid
 * re-indexing of the feature space for both model and data points. The behavior is simply no-op for
 * most operations and always outputs 0 for scoring. Those features with such function should be
 * deleted before final model persistence onto the disk.
 */
public class Zero implements Function {
  @Override
  public Function aggregate(Iterable<Function> functions, float scale, int numBins) {
    return this;
  }

  @Override
  public float evaluate(float... x) {
    return 0;
  }

  @Override
  public float evaluate(List<Double> values) {
    return 0;
  }

  @Override
  public void update(float delta, float... values) {

  }

  @Override
  public void update(float delta, List<Double> values) {

  }

  @Override
  public ModelRecord toModelRecord(String featureFamily, String featureName) {
    throw new IllegalAccessError("Zero point should never be persisted. Please delete or skip this feature instead.");
  }

  @Override
  public void setPriors(float[] params) {

  }

  @Override
  public void LInfinityCap(float cap) {

  }

  @Override
  public float LInfinityNorm() {
    return 0;
  }

  @Override
  public void resample(int newBins) {

  }

  @Override
  public double smooth(double tolerance, boolean toleranceIsPercentage) {
    return 0;
  }

  @Override
  public Function clone() throws CloneNotSupportedException {
    return this;
  }
}
