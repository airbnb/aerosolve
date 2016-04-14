package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.ModelRecord;
import com.google.common.primitives.Doubles;
import java.util.ArrayList;
import java.util.List;

// A piecewise linear spline implementation supporting updates.
public class Spline extends AbstractFunction {
  private static final long serialVersionUID = 5166347177557768302L;

  private int numBins;
  private double scale;
  private double binSize;
  private double binScale;

  public Spline(double minVal, double maxVal, double[] weights) {
    setupSpline(minVal, maxVal, weights);
  }

  public Spline(double minVal, double maxVal, int numBins) {
    if (maxVal <= minVal) {
      maxVal = minVal + 1.0f;
    }
    setupSpline(minVal, maxVal, new double[numBins]);
  }

  /*
    Generates new weights[] from numBins
   */
  public double[] weightsByNumBins(int numBins) {
    if (numBins == this.numBins) {
      return weights;
    } else {
      return newWeights(numBins);
    }
  }

  private double[] newWeights(int numBins) {
    assert (numBins != this.numBins);
    double[] newWeights = new double[numBins];
    double scale = 1.0d / (numBins - 1.0d);
    double diff = maxVal - minVal;
    for (int i = 0; i < numBins; i++) {
      double t = i * scale;
      double x = diff * t + minVal;
      newWeights[i] = evaluate(x);
    }
    return newWeights;
  }

  // A constructor from model record
  public Spline(ModelRecord record) {
    this.minVal = record.getMinVal();
    this.maxVal = record.getMaxVal();
    List<Double> weightVec = record.getWeightVector();
    this.numBins = weightVec.size();
    this.weights = new double[this.numBins];
    for (int j = 0; j < numBins; j++) {
      this.weights[j] = weightVec.get(j);
    }
    double diff = Math.max(maxVal - minVal, 1e-10d);
    this.scale = 1.0d / diff;
    this.binSize = diff / (numBins - 1.0d);
    this.binScale = 1.0d / binSize;
  }

  private void setupSpline(double minVal, double maxVal, double[] weights) {
    this.weights = weights;
    this.numBins = weights.length;
    this.minVal = minVal;
    this.maxVal = maxVal;
    double diff = Math.max(maxVal - minVal, 1e-10d);
    this.scale = 1.0d / diff;
    this.binSize = diff / (numBins - 1.0d);
    this.binScale = 1.0d / binSize;
  }

  @Override
  public Function aggregate(Iterable<Function> functions, double scale, int numBins) {
    int length = weights.length;
    double[] aggWeights = new double[length];

    for (Function fun : functions) {
      Spline spline = (Spline) fun;
      double[] w = spline.weightsByNumBins(numBins);
      for (int i = 0; i < length; i++) {
        aggWeights[i] += scale * w[i];
      }
    }
    return new Spline(minVal, maxVal, aggWeights);
  }

  @Override
  public double evaluate(double... x) {
    int bin = getBin(x[0]);
    if (bin == numBins - 1) {
      return weights[numBins - 1];
    }
    double t = getBinT(x[0], bin);
    return (1.0f - t) * weights[bin] + t * weights[bin + 1];
  }

  @Override
  public void update(double delta, double... values) {
    double x = values[0];
    int bin = getBin(x);
    if (bin == numBins - 1) {
      weights[numBins - 1] += delta;
    } else {
      double t = getBinT(x, bin);
      t = Math.max(0.0d, Math.min(1.0d, t));
      weights[bin] += (1.0d - t) * delta;
      weights[bin + 1] += t * delta;
    }
  }

  @Override
  public ModelRecord toModelRecord(String featureFamily, String featureName) {
    ModelRecord record = new ModelRecord();
    record.setFunctionForm(FunctionForm.SPLINE);
    record.setFeatureFamily(featureFamily);
    record.setFeatureName(featureName);
    ArrayList<Double> arrayList = new ArrayList<>();
    for (double weight : weights) {
      arrayList.add(weight);
    }
    record.setWeightVector(arrayList);
    record.setMinVal(minVal);
    record.setMaxVal(maxVal);
    return record;
  }

  @Override
  public void resample(int newBins) {
    if (newBins != numBins) {
      setupSpline(minVal, maxVal, newWeights(newBins));
    }
  }

  // Returns the lower bound bin
  public int getBin(double x) {
    int bin = (int) ((x - minVal) * scale * (numBins - 1));
    bin = Math.max(0, Math.min(numBins - 1, bin));
    return bin;
  }

  // Returns the t value in the bin (0, 1)
  public double getBinT(double x, int bin) {
    double lowerX = bin * binSize + minVal;
    double t = (x - lowerX) * binScale;
    t = Math.max(0.0d, Math.min(1.0d, t));
    return t;
  }

  public double L1Norm() {
    double sum = 0.0d;
    for (double weight : weights) {
      sum += Math.abs(weight);
    }
    return sum;
  }

  @Override
  public double LInfinityNorm() {
    return Math.max(Doubles.max(weights), Math.abs(Doubles.min(weights)));
  }

  @Override
  public void LInfinityCap(double cap) {
    if (cap <= 0.0d) return;
    double currentNorm = this.LInfinityNorm();
    if (currentNorm > cap) {
      double scale = cap / currentNorm;
      for (int i = 0; i < weights.length; i++) {
        weights[i] *= scale;
      }
    }
  }

  @Override
  public void setPriors(double[] params) {
    double start = params[0];
    double end = params[1];
    // fit a line based on the input starting weight and ending weight
    for (int i = 0; i < numBins; i++) {
      double t = i / (numBins - 1.0d);
      weights[i] = ((1.0d - t) * start + t * end);
    }
  }

  @Override
  public void smooth(double tolerance) {
    FunctionUtil.smooth(tolerance, weights);
  }
}
