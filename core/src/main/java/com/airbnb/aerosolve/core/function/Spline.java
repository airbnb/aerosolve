package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.ModelRecord;

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
  
  // A resampling constructor
  public Spline(Spline other, int newBins) {
    setupSpline(other.minVal, other.maxVal, other.weightsByNumBins(newBins, true));
  }

  /*
    Generates new weights[] from numBins
   */
  public float[] weightsByNumBins(int numBins, boolean clone) {
    if (numBins == this.numBins) {
      if (clone) {
        return weights.clone();
      } else {
        return weights;
      }
    } else {
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
      float[] w = spline.weightsByNumBins(numBins, false);
      for (int i = 0; i < length; i++) {
        aggWeights[i] += scale * w[i];
      }
    }
    return new Spline(minVal, maxVal, aggWeights);
  }

  @Override
  public float evaluate(float x) {
    int bin = getBin(x);
    if (bin == numBins - 1) {
      return weights[numBins - 1];
    }
    float t = getBinT(x, bin);
    t = Math.max(0.0f, Math.min(1.0f, t));
    float result = (1.0f - t) * weights[bin] + t * weights[bin + 1];
    return result;
  }

  @Override
  public void update(float x, float delta) {
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
    record.setFunctionForm(FunctionForm.SPLINE);
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

  public void resample(int newBins) {
    float[] newWeights = new float[newBins];
    if (newBins != numBins) {
      float scale = 1.0f / (newBins - 1.0f);
      float diff = maxVal - minVal;
      for (int i = 0; i < newBins; i++) {
        float t = i * scale;
        float x = diff * t + minVal;
        newWeights[i] = evaluate(x);
      }
      setupSpline(minVal, maxVal, newWeights);
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
    float best = 0.0f;
    for (int i = 0; i < weights.length; i++) {
      best = Math.max(best, Math.abs(weights[i]));
    }
    return best;
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
  public void setPriors(String[] params) {
    float start = Float.valueOf(params[2]);
    float end = Float.valueOf(params[3]);
    // fit a line based on the input starting weight and ending weight
    for (int i = 0; i < numBins; i++) {
      float t = i / (numBins - 1.0f);
      weights[i] = ((1.0f - t) * start + t * end);
    }
  }
}
