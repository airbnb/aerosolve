package com.airbnb.aerosolve.core.functions;

import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.util.Util;
import java.lang.reflect.Constructor;
import java.util.Map;
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
  protected double[] weights;
  @Getter
  protected double minVal;
  @Getter
  protected double maxVal;

  private static Map<String, Constructor<? extends AbstractFunction>> FUNCTION_CONSTRUCTORS;

  @Override
  public String toString() {
    return String.format("minVal=%f, maxVal=%f, weights=%s",
        minVal, maxVal, Arrays.toString(weights));
  }

  public static Function buildFunction(ModelRecord record) {
    FunctionForm funcForm = record.getFunctionForm();
    if (FUNCTION_CONSTRUCTORS == null) {
      loadFunctionConstructors();
    }
    String name = funcForm.name().toLowerCase();
    Constructor<? extends AbstractFunction> constructor =
        FUNCTION_CONSTRUCTORS.get(funcForm.name().toLowerCase());
    if (constructor == null) {
      throw new IllegalArgumentException(
          String.format("No function exists with name %s", name));
    }
    try {
      return constructor.newInstance(record);
    } catch (InstantiationException | IllegalAccessException | InvocationTargetException e) {
      throw new IllegalStateException(
          String.format("There was an error instantiating Function of type %s : %s",
                        name, e.getMessage()), e);
    }
  }

  private static synchronized void loadFunctionConstructors() {
    if (FUNCTION_CONSTRUCTORS != null) {
      return;
    }
    FUNCTION_CONSTRUCTORS = Util.loadConstructorsFromPackage(AbstractFunction.class,
                                                             "com.airbnb.aerosolve.core.functions",
                                                             "",
                                                             ModelRecord.class);
  }
}
