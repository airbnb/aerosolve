package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.ModelRecord;
import com.google.common.primitives.Floats;

import java.util.ArrayList;
import java.util.List;

// A piecewise linear spline implementation supporting updates.
public class Spline extends AbstractFunction {
  private static final long serialVersionUID = 5166347177557768302L;

  private int numBins;
  private float scale;
  private float binSize;
  private float binScale;

  public Spline(float minVal, float maxVal, float [] weights) {
    setupSpline(minVal, maxVal, weights);
  }

  public Spline(float minVal, float maxVal, int numBins) {
    if (maxVal <= minVal) {
      maxVal = minVal + 1.0f;
    }
    setupSpline(minVal, maxVal, new float[numBins]);
  }

  /*
    Generates new weights[] from numBins
   */
  public float[] weightsByNumBins(int numBins) {
    if (numBins == this.numBins) {
      return weights;
    } else {
      return newWeights(numBins);
    }
  }

  private float[] newWeights(int numBins) {
    assert (numBins != this.numBins);
    float[] newWeights = new float[numBins];
    float scale = 1.0f / (numBins - 1.0f);
    float diff = maxVal - minVal;
    for (int i = 0; i < numBins; i++) {
      float t = i * scale;
      float x = diff * t + minVal;
      newWeights[i] = evaluate(x);
    }
    return newWeights;
  }

  // A constructor from model record
  public Spline(ModelRecord record) {
    this.minVal = (float) record.getMinVal();
    this.maxVal = (float) record.getMaxVal();
    List<Double> weightVec = record.getWeightVector();
    this.numBins = weightVec.size();
    this.weights = new float[this.numBins];
    for (int j = 0; j < numBins; j++) {
      this.weights[j] = weightVec.get(j).floatValue();
    }
    float diff = Math.max(maxVal - minVal, 1e-10f);
    this.scale = 1.0f / diff;
    this.binSize = diff / (numBins - 1.0f);
    this.binScale = 1.0f / binSize;
  }
  
  private void setupSpline(float minVal, float maxVal, float [] weights) {
    this.weights = weights;
    this.numBins = weights.length;
    this.minVal = minVal;
    this.maxVal = maxVal;
    float diff = Math.max(maxVal - minVal, 1e-10f);
    this.scale = 1.0f / diff;
    this.binSize = diff / (numBins - 1.0f);
    this.binScale = 1.0f / binSize;
  }

  @Override
  public Function aggregate(Iterable<Function> functions, float scale, int numBins) {
    int length = weights.length;
    float[] aggWeights = new float[length];

    for (Function fun: functions) {
      Spline spline = (Spline) fun;
      float[] w = spline.weightsByNumBins(numBins);
      for (int i = 0; i < length; i++) {
        aggWeights[i] += scale * w[i];
      }
    }
    return new Spline(minVal, maxVal, aggWeights);
  }

  @Override
  public float evaluate(float ... x) {
    int bin = getBin(x[0]);
    if (bin == numBins - 1) {
      return weights[numBins - 1];
    }
    float t = getBinT(x[0], bin);
    t = Math.max(0.0f, Math.min(1.0f, t));
    float result = (1.0f - t) * weights[bin] + t * weights[bin + 1];
    return result;
  }

  @Override
  public void update(float delta, float ... values) {
    float x = values[0];
    int bin = getBin(x);
    if (bin == numBins - 1) {
      weights[numBins - 1] += delta;
    } else {
      float t = getBinT(x, bin);
      t = Math.max(0.0f, Math.min(1.0f, t));
      weights[bin] += (1.0f - t) * delta;
      weights[bin + 1] += t * delta;
    }
  }

  @Override
  public ModelRecord toModelRecord(String featureFamily, String featureName) {
    ModelRecord record = new ModelRecord();
    record.setFunctionForm(FunctionForm.Spline);
    record.setFeatureFamily(featureFamily);
    record.setFeatureName(featureName);
    ArrayList<Double> arrayList = new ArrayList<Double>();
    for (int i = 0; i < weights.length; i++) {
      arrayList.add((double) weights[i]);
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
  public int getBin(float x) {
    int bin = (int) Math.floor((x - minVal) * scale * (numBins - 1));
    bin = Math.max(0, Math.min(numBins - 1, bin));
    return bin;
  }

  // Returns the t value in the bin (0, 1)
  public float getBinT(float x, int bin) {
    float lowerX = bin * binSize + minVal;
    float t = (x - lowerX) * binScale;
    t = Math.max(0.0f, Math.min(1.0f, t));
    return t;
  }

  public float L1Norm() {
    float sum = 0.0f;
    for (int i = 0; i < weights.length; i++) {
      sum += Math.abs(weights[i]);
    }
    return sum;
  }

  @Override
  public float LInfinityNorm() {
    return Math.max(Floats.max(weights), Math.abs(Floats.min(weights)));
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
  public void setPriors(float[] params) {
    float start = params[0];
    float end = params[1];
    // fit a line based on the input starting weight and ending weight
    for (int i = 0; i < numBins; i++) {
      float t = i / (numBins - 1.0f);
      weights[i] = ((1.0f - t) * start + t * end);
    }
  }
}
