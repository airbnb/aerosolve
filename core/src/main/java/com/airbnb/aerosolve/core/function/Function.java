package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.ModelRecord;

import java.io.Serializable;

public interface Function extends Serializable {

  FunctionForm getFunctionForm();

  // TODO rename numBins to something else, since it's a Spline specific thing
  Function aggregate(Iterable<Function> functions, float scale, int numBins);

  float evaluate(float x);

  void update(float x, float delta);

  ModelRecord toModelRecord(String featureFamily, String featureName);

  void setPriors(String[] params);

  void LInfinityCap(float cap);

  float LInfinityNorm();
}
