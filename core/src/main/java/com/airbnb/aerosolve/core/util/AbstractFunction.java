package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.FunctionForm;
import lombok.Getter;
import lombok.Setter;

import java.io.Serializable;

/**
 * Base class for functions
 */
abstract public class AbstractFunction implements Serializable{

  @Getter @Setter
  protected FunctionForm functionForm; // default function is Spline

  @Getter @Setter
  private float[] weights;

  @Getter
  private float minVal;

  @Getter
  private float maxVal;

  abstract public AbstractFunction makeCopy();

  abstract public float evaluate(float x);

  abstract public void update(float x, float delta);

  abstract public ModelRecord toModelRecord(String featureFamily, String featureName);

  abstract public void setPriors(float[] params);

  abstract public void LInfinityCap(float cap);

  abstract public float LInfinityNorm();
}
