package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.ModelRecord;
import lombok.Getter;

import java.util.Arrays;
import java.util.List;

/**
 * Base class for functions
 */
public abstract class AbstractFunction implements Function {
  @Getter
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

  @Override
  public float evaluate(List<Double> values) {
    throw new RuntimeException("method not implemented");
  }

  @Override
  public void update(float delta, List<Double> values){
    throw new RuntimeException("method not implemented");
  }

  public static Function buildFunction(ModelRecord record) {
    FunctionForm funcForm = record.getFunctionForm();
    try {
      return (Function) Class.forName("com.airbnb.aerosolve.core.function." +
          funcForm.name()).getDeclaredConstructor(ModelRecord.class).newInstance(record);
    } catch (Exception e) {
      e.printStackTrace();
    }
    throw new RuntimeException("unable to decode " + funcForm.name());
  }

  @Override
  public AbstractFunction clone() throws CloneNotSupportedException {
    return (AbstractFunction) super.clone();
  }
}
