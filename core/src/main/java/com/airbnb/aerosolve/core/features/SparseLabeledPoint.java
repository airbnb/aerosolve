package com.airbnb.aerosolve.core.features;

import lombok.Data;

import java.io.Serializable;

/**
 * This is a compressed representation of a feature vector indexed via a feature index map. It
 * captures values in flat representation for space and runtime efficiency during model training.
 */
@Data
public class SparseLabeledPoint implements Serializable {
  public final boolean isTraining;
  public final double label;
  public final int[] indices;
  public final float[] values;
  public final int[] denseIndices;
  public final float[][] denseValues;
}
