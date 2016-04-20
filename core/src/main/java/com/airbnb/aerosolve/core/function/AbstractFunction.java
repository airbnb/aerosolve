package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.FunctionForm;
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
  protected FunctionForm functionForm = FunctionForm.SPLINE;
  @Getter
  protected float minVal;
  @Getter
  protected float maxVal;

  @Override
  public Function aggregate(Iterable<Function> functions, float scale, int numBins) {
    int length = weights.length;
    float[] aggWeights = new float[length];

    for (Function fun: functions) {
      AbstractFunction abstractFunction = (AbstractFunction) fun;
      for (int i = 0; i < length; i++) {
        aggWeights[i] += scale * abstractFunction.weights[i];
      }
    }
    return new Linear(minVal, maxVal, aggWeights);
  }

  @Override
  public String toString() {
    return String.format("minVal=%f, maxVal=%f, weights=%s",
        minVal, maxVal, Arrays.toString(weights));
  }
}
