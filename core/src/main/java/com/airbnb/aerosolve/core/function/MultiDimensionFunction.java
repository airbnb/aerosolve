package com.airbnb.aerosolve.core.function;

import java.io.Serializable;
import java.util.List;

public interface MultiDimensionFunction extends Serializable {

  double evaluate(List<Double> coordinates);

  void update(List<Double> coordinates, double delta);

}
