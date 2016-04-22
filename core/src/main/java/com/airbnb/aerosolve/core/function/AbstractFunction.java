package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.ModelRecord;
import lombok.Getter;
import lombok.Setter;

import java.lang.reflect.InvocationTargetException;
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

  public static Function buildFunction(ModelRecord record) {
    FunctionForm funcForm = record.getFunctionForm();
    try {
      return (Function) Class.forName("com.airbnb.aerosolve.core.function." +
          funcForm.name()).getDeclaredConstructor(ModelRecord.class).newInstance(record);
    } catch (InstantiationException e) {
      e.printStackTrace();
    } catch (IllegalAccessException e) {
      e.printStackTrace();
    } catch (InvocationTargetException e) {
      e.printStackTrace();
    } catch (NoSuchMethodException e) {
      e.printStackTrace();
    } catch (ClassNotFoundException e) {
      e.printStackTrace();
    }
    throw new RuntimeException("no such function " + funcForm.name());
  }
}
