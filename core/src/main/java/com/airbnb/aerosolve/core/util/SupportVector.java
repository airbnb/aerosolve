package com.airbnb.aerosolve.core.util;

import lombok.Getter;
import lombok.Setter;

import java.io.Serializable;
import java.util.Random;

import com.airbnb.aerosolve.core.FunctionForm;

public class SupportVector implements Serializable {
  // Dense support vector value.
  @Getter @Setter
  FloatVector floatVector;
 
  // What kind of kernel e.g. RBF
  @Getter @Setter
  FunctionForm form;
  
  // Kernel parameter .. for:
  // RBF : response = exp(- scale * || sv - other || ^ 2) 
  @Getter @Setter
  float scale;
  
  SupportVector(FloatVector fv, FunctionForm f, float s) {
    floatVector = fv;
    form = f;
    scale = s;
  }
  
  // Evaluates the support vector with the other.
  public float evaluate(FloatVector other) {
    float result = 0.0f;
    switch (form) {
      case RADIAL_BASIS_FUNCTION: {
        result = (float) Math.exp(-scale * floatVector.l2Distance2(other));
        break;
      }
    }
    return result;
  }
  
}