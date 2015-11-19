package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.ModelRecord;
import lombok.Getter;
import lombok.Setter;

import java.io.Serializable;

/**
 * Base class for functions
 */
abstract public class AbstractFunction implements Serializable{
  @Getter @Setter
  protected String functionForm; // default function is Spline

  @Getter @Setter
  private float[] weights;

  abstract public float evaluate(float x);

  abstract public ModelRecord toModelRecord(String featureFamily, String featureName);

}
