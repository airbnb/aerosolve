package com.airbnb.aerosolve.core.functions;

import com.airbnb.aerosolve.core.ModelRecord;

import java.io.Serializable;

public interface Function extends Serializable {
  // TODO rename numBins to something else, since it's a Spline specific thing
  Function aggregate(Iterable<Function> functions, double scale, int numBins);

  double evaluate(double ... x);

  void update(double delta, double ... values);

  ModelRecord toModelRecord(String featureFamily, String featureName);

  void setPriors(double[] params);

  void LInfinityCap(double cap);

  double LInfinityNorm();

  void resample(int newBins);

  void smooth(double tolerance);
}
