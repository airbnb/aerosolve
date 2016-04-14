package com.airbnb.aerosolve.core.functions;

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

  public Linear(double minVal, double maxVal) {
    this(minVal, maxVal, new double[2]);
  }

  public Linear(double minVal, double maxVal, double[] weights) {
    this.weights = weights;
    this.minVal = minVal;
    this.maxVal = maxVal;
  }

  @Override
  public Function aggregate(Iterable<Function> functions, double scale, int numBins) {
    int length = weights.length;
    double[] aggWeights = new double[length];

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
    weights = new double[2];
    for (int j = 0; j < Math.min(n, 2); j++) {
      weights[j] = weightVec.get(j);
    }
    minVal = record.getMinVal();
    maxVal = record.getMaxVal();
  }

  @Override
  public void update(double delta, double... values) {
    weights[0] += delta;
    weights[1] += delta * normalization(values[0]);
  }

  @Override
  public void setPriors(double[] params) {
    weights[0] = params[0];
    weights[1] = params[1];
  }

  public double evaluate(double... x) {
    return weights[0] + weights[1] * normalization(x[0]);
  }

  @Override
  public ModelRecord toModelRecord(String featureFamily, String featureName) {
    ModelRecord record = new ModelRecord();
    record.setFunctionForm(FunctionForm.LINEAR);
    record.setFeatureFamily(featureFamily);
    record.setFeatureName(featureName);
    record.setMinVal(minVal);
    record.setMaxVal(maxVal);
    ArrayList<Double> arrayList = new ArrayList<>();
    arrayList.add(weights[0]);
    arrayList.add(weights[1]);
    record.setWeightVector(arrayList);
    return record;
  }

  @Override
  public void LInfinityCap(double cap) {
    if (cap <= 0.0f) return;
    double currentNorm = this.LInfinityNorm();
    if (currentNorm > cap) {
      double scale = cap / currentNorm;
      for (int i = 0; i < weights.length; i++) {
        weights[i] *= scale;
      }
    }
  }

  @Override
  public double LInfinityNorm() {
    double f0 = weights[0];
    double f1 = weights[0] + weights[1];
    return Math.max(Math.abs(f0), Math.abs(f1));
  }

  private double normalization(double x) {
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
  public void smooth(double tolerance) {
  }
}
