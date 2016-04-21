package com.airbnb.aerosolve.core.function;

import lombok.Getter;
import lombok.Setter;

import java.util.Arrays;

/**
 * Base class for functions
 */
public abstract class AbstractFunction implements Function {
  @Getter
  @Setter
  protected float[] weights;
  @Getter
  protected float minVal;
  @Getter
  protected float maxVal;

  @Override
  public String toString() {
    return String.format("minVal=%f, maxVal=%f, weights=%s",
        minVal, maxVal, Arrays.toString(weights));
  }
}
