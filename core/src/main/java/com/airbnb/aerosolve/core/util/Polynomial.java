package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.ModelRecord;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;

/**
 * Polynomial function
 */
public class Polynomial extends AbstractFunction {
  @Getter @Setter
  private float[] weights;

  public Polynomial(float [] weights) {
    this.weights = weights;
  }

  public Polynomial(ModelRecord record) {
    List<Double> weightVec = record.getWeightVector();
    int n = weightVec.size();
    this.weights = new float[n];
    for (int j = 0; j < n; j++) {
      this.weights[j] = weightVec.get(j).floatValue();
    }
  }

  @Override
  public float evaluate(float x) {
    int n = weights.length;
    float result = 0.0f;
    for (int i = 0; i < n; i++) {
      result += weights[i] * Math.pow(x, i);
    }
    return result;
  }

  @Override
  public ModelRecord toModelRecord(String featureFamily, String featureName) {
    ModelRecord record = new ModelRecord();
    record.setFunctionForm("Polynomial");
    record.setFeatureFamily(featureFamily);
    record.setFeatureName(featureName);
    ArrayList<Double> arrayList = new ArrayList<Double>();
    for (int i = 0; i < weights.length; i++) {
      arrayList.add((double) weights[i]);
    }
    record.setWeightVector(arrayList);
    return record;
  }
}