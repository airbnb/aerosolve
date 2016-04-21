package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.ModelRecord;

import java.io.Serializable;

public interface Function extends Serializable {

  FunctionForm getFunctionForm();

  // TODO rename numBins to something else, since it's a Spline specific thing
  Function aggregate(Iterable<Function> functions, float scale, int numBins);

  // TODO remove setWeights
  void setWeights(float[] weights);

  float getMinVal();

  float getMaxVal();

  Function makeCopy();

  float evaluate(float x);

  void update(float x, float delta);

  ModelRecord toModelRecord(String featureFamily, String featureName);

  void setPriors(float[] params);

  void LInfinityCap(float cap);

  float LInfinityNorm();
}
