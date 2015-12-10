package com.airbnb.aerosolve.core.util;

import lombok.Getter;
import lombok.Setter;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.ModelRecord;

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
  
  // Weight of the kernel
  @Getter @Setter
  float weight;

  public SupportVector(FloatVector fv, FunctionForm f, float s, float wt) {
    floatVector = fv;
    form = f;
    scale = s;
    weight = wt;
  }

  public SupportVector(ModelRecord rec) {
    scale = (float) rec.scale;
    form = rec.getFunctionForm();
    weight = (float) rec.getFeatureWeight();
    int size = rec.weightVector.size();
    floatVector = new FloatVector(size);
    for (int i = 0; i < size; i++) {
      floatVector.getValues()[i] = rec.weightVector.get(i).floatValue();
    }
  }

  public ModelRecord toModelRecord() {
    ModelRecord rec = new ModelRecord();
    rec.setScale(scale);
    rec.setFunctionForm(form);
    ArrayList<Double> weightVector = new ArrayList<>();
    for (int i = 0; i < floatVector.getValues().length; i++) {
      weightVector.add((double) floatVector.getValues()[i]);
    }
    rec.setWeightVector(weightVector);
    rec.setFeatureWeight(weight);
    return rec;
  }
  
  // Evaluates the support vector with the other.
  public float evaluateUnweighted(FloatVector other) {
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
        double cos_theta = Math.max(0.0, Math.min(1.0, top / Math.sqrt(bot)));
        double theta = Math.acos(cos_theta);
        result = (float) (1.0 - theta / Math.PI);
        break;
      }
    }
    return result;
  }
  
  // Evaluates the weighted support vector
  public float evaluate(FloatVector other) {
    return weight * evaluateUnweighted(other);
  }
  
}