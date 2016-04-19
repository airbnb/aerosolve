package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.ModelRecord;

import java.io.Serializable;

public interface Function extends Serializable {

  FunctionForm getFunctionForm();

  float[] getWeights();

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
