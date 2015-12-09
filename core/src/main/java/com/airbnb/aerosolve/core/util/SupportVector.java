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
      case ARC_COSINE: {
        float top = floatVector.dot(other);
        float bot = floatVector.dot(floatVector) * other.dot(other);
        if (bot < 1e-6f) {
          bot = 1e-6f;
        }
        double cos_theta = top / Math.sqrt(bot);
        double theta = Math.acos(cos_theta);
        result = (float) (1.0 - theta / Math.PI);
        break;
      }
    }
    return result;
  }
  
}