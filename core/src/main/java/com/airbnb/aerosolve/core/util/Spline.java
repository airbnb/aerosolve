package com.airbnb.aerosolve.core.util;

import lombok.Getter;
import lombok.Setter;

import java.io.Serializable;
import java.util.Map;
import java.util.TreeMap;

// A piecewise linear spline implementation supporting updates.
public class Spline implements Serializable {
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
    this.weights = weights;
    this.numBins = weights.length;
    this.minVal = minVal;
    this.maxVal = maxVal;
    float diff = Math.max(maxVal - minVal, 1e-10f);
    this.scale = 1.0f / diff;
    this.binSize = diff / (numBins - 1.0f);
    this.binScale = 1.0f / binSize;
  }

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

}
