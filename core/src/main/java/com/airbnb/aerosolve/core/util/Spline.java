package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.ModelRecord;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;
import java.io.Serializable;
import java.util.Map;
import java.util.TreeMap;

// A piecewise linear spline implementation supporting updates.
public class Spline extends AbstractFunction {
  private static final long serialVersionUID = 5166347177557768302L;

  @Getter @Setter
  private float[] weights;

  private int numBins;
  @Getter
  private float minVal;
  @Getter
  private float maxVal;
  private float scale;
  private float binSize;
  private float binScale;

  public Spline(float minVal, float maxVal, float [] weights) {
    setupSpline(minVal, maxVal, weights);
  }
  
  // A resampling constructor
  public Spline(Spline other, int newBins) {
    float[] newWeights = new float[newBins];
    if (newBins == other.numBins) {
      for (int i = 0; i < newBins; i++) {
        newWeights[i] = other.weights[i];
      }
    } else {
      float scale = 1.0f / (newBins - 1.0f);
      float diff = other.maxVal - other.minVal;
      for (int i = 0; i < newBins; i++) {
        float t = i * scale;
        float x = diff * t + other.minVal;
        newWeights[i] = other.evaluate(x);
      }
    }
    setupSpline(other.minVal, other.maxVal, newWeights);
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
    record.setFunctionForm("Spline");
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

  public float LInfinityNorm() {
    float best = 0.0f;
    for (int i = 0; i < weights.length; i++) {
      best = Math.max(best, Math.abs(weights[i]));
    }
    return best;
  }

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
}
