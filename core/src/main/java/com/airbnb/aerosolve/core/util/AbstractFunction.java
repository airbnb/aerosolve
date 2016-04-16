package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.ModelRecord;

import java.io.Serializable;

/**
 * Base class for functions
 */
public interface AbstractFunction extends Serializable{
  FunctionForm getFunctionForm();
  float[] getWeights();
  // TODO remove setWeights
  void setWeights(float[] weights);
  float getMinVal();
  float getMaxVal();
  AbstractFunction makeCopy();

  float evaluate(float x);

  void update(float x, float delta);

  ModelRecord toModelRecord(String featureFamily, String featureName);

  void setPriors(float[] params);

  void LInfinityCap(float cap);

  float LInfinityNorm();
}
