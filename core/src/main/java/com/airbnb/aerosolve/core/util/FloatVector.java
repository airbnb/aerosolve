package com.airbnb.aerosolve.core.util;

import lombok.Getter;
import lombok.Setter;

import java.io.Serializable;
import java.util.Random;

public class FloatVector implements Serializable {
  private static final Random rnd = new java.util.Random();

  public static class MinMaxResult {
    public int minIndex;
    public int maxIndex;
    public float minValue;
    public float maxValue;
  }

  @Getter @Setter
  public float values[];
  public FloatVector() {
  }

  public FloatVector(int num) {
    setZero(num);
  }

  public FloatVector(float val[]) {
    values = val;
  }

  public static FloatVector getGaussianVector(int num) {
    float[] init = new float[num];
    double mult = 1.0 / Math.sqrt(num);
    for (int i = 0; i < num; i++) {
      init[i] = (float) (rnd.nextGaussian() * mult);
    }
    return new FloatVector(init);
  }

  public void setZero(int num) {
    values = new float[num];
  }

  public void setRandom(int num, float scale) {
    setZero(num);
    for (int i = 0; i < values.length; i++) {
      values[i] = (2.0f * rnd.nextFloat() - 1.0f) * scale;
    }
  }

  public float dot(FloatVector other) {
    assert(values.length == other.values.length);
    float sum = 0;
    for (int i  = 0; i < values.length; i++) {
      sum += values[i] * other.values[i];
    }
    return sum;
  }

  public void add(FloatVector other) {
    assert(values.length == other.values.length);
    for (int i  = 0; i < values.length; i++) {
      values[i] += other.values[i];
    }
  }

  public void multiplyAdd(float w, FloatVector other) {
    assert(values.length == other.values.length);
    for (int i  = 0; i < values.length; i++) {
      values[i] += w * other.values[i];
    }
  }

  public void rectify() {
    for (int i = 0; i < values.length; i++) {
      if (values[i] < 0) {
        values[i] = 0.0f;
      }
    }
  }

  public String toString() {
    return java.util.Arrays.toString(values);
  }

  public static FloatVector Hadamard(FloatVector a, FloatVector b) {
    assert(a.values.length == b.values.length);
    FloatVector out = new FloatVector(a.values.length);
    for (int i = 0; i < a.values.length; i++) {
      out.values[i] = a.values[i] * b.values[i];
    }
    return out;
  }

  // Returns the indices and values of the min/max of the vector.
  // The first half of the vector stores the "min" hidden layer
  // and the second half of the vector stores the "max".
  // In both cases the largest value is used as a selector, the name
  // is just used to keep the positive and negative maxout planes
  // disjoint but for memory efficiency stored in the same vector.
  public MinMaxResult getMinMaxResult() {
    MinMaxResult result = new MinMaxResult();
    int mid = values.length / 2;
    result.minIndex = 0;
    result.minValue = values[0];
    for (int i = 1; i < mid; i++) {
      float curr = values[i];
      if (curr > result.minValue) {
        result.minValue = curr;
        result.minIndex = i;
      }
    }
    result.maxIndex = 1;
    result.maxValue = values[mid];
    for (int i = mid + 1; i < values.length; i++) {
      float curr = values[i];
      if (curr > result.maxValue) {
        result.maxValue = curr;
        result.maxIndex = i;
      }
    }
    return result;
  }
}