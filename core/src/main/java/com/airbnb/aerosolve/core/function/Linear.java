package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.ModelRecord;

import java.util.ArrayList;
import java.util.List;

/**
 * Linear function f(x) = weights[1]*x+weights[0]
 */
public class Linear extends AbstractFunction {
  // weights[0] is offset, weights[1] is slope
  public Linear(Linear other) {
    weights = other.weights.clone();
    minVal = other.getMinVal();
    maxVal = other.getMaxVal();
  }

  public Linear(float minVal, float maxVal) {
    this(minVal, maxVal, new float[2]);
  }

  public Linear(float minVal, float maxVal, float[] weights) {
    this.weights = weights;
    this.minVal = minVal;
    this.maxVal = maxVal;
  }

  @Override
  public Function aggregate(Iterable<Function> functions, float scale, int numBins) {
    int length = weights.length;
    float[] aggWeights = new float[length];

    for (Function fun: functions) {
      Linear linear = (Linear) fun;
      for (int i = 0; i < length; i++) {
        aggWeights[i] += scale * linear.weights[i];
      }
    }
    return new Linear(minVal, maxVal, aggWeights);
  }

  public Linear(ModelRecord record) {
    List<Double> weightVec = record.getWeightVector();
    int n = weightVec.size();
    weights = new float[2];
    for (int j = 0; j < Math.min(n, 2); j++) {
      weights[j] = weightVec.get(j).floatValue();
    }
    minVal = (float) record.getMinVal();
    maxVal = (float) record.getMaxVal();
  }

  @Override
  public void update(float delta, float ... values) {
    weights[0] += delta;
    weights[1] += delta * normalization(values[0]);
  }

  @Override
  public void setPriors(float[] params) {
    weights[0] = params[0];
    weights[1] = params[1];
  }

  @Override
  public float evaluate(float ... x) {
    return weights[0] + weights[1] * normalization(x[0]);
  }

  @Override
  public ModelRecord toModelRecord(String featureFamily, String featureName) {
    ModelRecord record = new ModelRecord();
    record.setFunctionForm(FunctionForm.Linear);
    record.setFeatureFamily(featureFamily);
    record.setFeatureName(featureName);
    record.setMinVal(minVal);
    record.setMaxVal(maxVal);
    ArrayList<Double> arrayList = new ArrayList<Double>();
    arrayList.add((double) weights[0]);
    arrayList.add((double) weights[1]);
    record.setWeightVector(arrayList);
    return record;
  }

  @Override
  public void LInfinityCap(float cap) {
    if (cap <= 0.0f) return;
    float currentNorm = this.LInfinityNorm();
    if (currentNorm > cap) {
      float scale = cap / currentNorm;
      for (int i = 0; i < weights.length; i++) {
        weights[i] *= scale;
      }
    }
  }

  @Override
  public float LInfinityNorm() {
    // return max absolute contribution
    float f0 = weights[0];
    float f1 = weights[0] + weights[1];
    return Math.max(Math.abs(f0), Math.abs(f1));
  }

  private float normalization(float x) {
    if (minVal < maxVal) {
      return (x - minVal) / (maxVal - minVal);
    } else if (minVal == maxVal && maxVal != 0){
      return x / maxVal;
    } else {
      return x;
    }
  }

  @Override
  public void resample(int newBins) {
  }

  @Override
  public double smooth(double tolerance, boolean toleranceIsPercentage) {
    return 0;
  }

  @Override
  public Linear clone() throws CloneNotSupportedException {
    Linear copy = (Linear) super.clone();
    copy.weights = weights.clone();
    return copy;
  }
}
