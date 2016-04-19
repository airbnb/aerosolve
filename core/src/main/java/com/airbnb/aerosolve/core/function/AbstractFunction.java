package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.FunctionForm;
import lombok.Getter;
import lombok.Setter;

/**
 * Base class for functions
 */
public abstract class AbstractFunction implements Function {
  @Getter
  @Setter
  protected float[] weights;
  @Getter
  protected FunctionForm functionForm = FunctionForm.SPLINE;
  @Getter
  protected float minVal;
  @Getter
  protected float maxVal;
}
