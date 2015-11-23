package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.FunctionForm;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;

/**
 * Linear function f(x) = weights[1]*x+weights[0]
 */
public class Linear extends AbstractFunction {
  // weights[0] is offset, weights[1] is slope
  @Getter @Setter
  private float[] weights;

  @Getter
  private FunctionForm functionForm = FunctionForm.LINEAR;

  public Linear() {
    this.weights = new float[2];
    // default function: f(x) = x
    this.weights[0] = 0.0f;
    this.weights[1] = 1.0f;
  }

  public Linear(Linear other) {
    this.weights = new float[2];
    this.weights[0] = other.getWeights()[0];
    this.weights[1] = other.getWeights()[1];
  }

  @Override
  public AbstractFunction makeCopy() {
    Linear linear = new Linear(this);
    return linear;
  }

  public Linear(float [] weights) {
    this.weights = weights;
  }

  public Linear(ModelRecord record) {
    List<Double> weightVec = record.getWeightVector();
    int n = weightVec.size();
    this.weights = new float[2];
    for (int j = 0; j < Math.min(n, 2); j++) {
      this.weights[j] = weightVec.get(j).floatValue();
    }
  }

  @Override
  public void update(float x, float delta) {
    weights[0] += delta;
    weights[1] += delta * x;
  }

  @Override
  public void setPriors(float[] params) {
    weights[0] = params[0];
    weights[1] = params[1];
  }

  @Override
  public float evaluate(float x) {
    return weights[0] + weights[1] * x;
  }

  @Override
  public ModelRecord toModelRecord(String featureFamily, String featureName) {
    ModelRecord record = new ModelRecord();
    record.setFunctionForm(FunctionForm.LINEAR);
    record.setFeatureFamily(featureFamily);
    record.setFeatureName(featureName);
    ArrayList<Double> arrayList = new ArrayList<Double>();
    arrayList.add((double) weights[0]);
    arrayList.add((double) weights[1]);
    record.setWeightVector(arrayList);
    return record;
  }

  @Override
  public void LInfinityCap(float... input) {
    float cap = input[0];
    float val = input[1];
    if (cap <= 0.0f) return;
    float y = this.evaluate(val);
    if (y > cap) {
      weights[0] = weights[0] - (y - cap);
    } else if (y < -cap) {
      weights[0] = weights[0] + (- y - cap);
    }
  }
}
