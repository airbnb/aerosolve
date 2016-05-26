package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.ModelRecord;

import java.io.Serializable;
import java.util.List;

public interface Function extends Serializable, Cloneable {
  // TODO rename numBins to something else, since it's a Spline specific thing
  Function aggregate(Iterable<Function> functions, float scale, int numBins);

  float evaluate(float ... x);
  // TODO change all float to double
  float evaluate(List<Double> values);

  void update(float delta, float ... values);
  void update(float delta, List<Double> values);

  ModelRecord toModelRecord(String featureFamily, String featureName);

  void setPriors(float[] params);

  void LInfinityCap(float cap);

  float LInfinityNorm();

  void resample(int newBins);

  // for function not support smooth, just return 0
  // for function support smooth, return errAndCoeff
  double smooth(double tolerance, boolean toleranceIsPercentage);

  Function clone() throws CloneNotSupportedException;
}
