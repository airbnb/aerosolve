package com.airbnb.aerosolve.core.images;

import java.awt.image.BufferedImage;
import java.lang.Override;
import java.lang.Math;

/*
 Creates a histogram of oriented gradients.
 http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
 */
public class HOGFeature extends ImageFeature {
  private static final int kNumBins = 9;
  // The histogram of oriented gradients.
  // Split into two parts : gradient magnitude and bin.
  private float[][] gradientMagnitude = null;
  private byte[][] gradientBin = null;
  private BufferedImage imgRef = null;
  // Lookup table so we do not have to compute the atan
  // millions of times.
  private static byte[][] atanTable = computeAtanTable();

  private static byte[][] computeAtanTable() {
    // The possible values of dx and dy range from
    // -255 to 255. So there are 511 values each.
    // To prevent divisions by zero
    final float kEpsilon = 1e-3f;
    byte[][] atanTable = new byte[511][511];
    for (int y = 0; y < 511; y++) {
      for (int x = 0; x < 511; x++) {
        float dx = x - 255;
        float dy = y - 255;
        if (x == 0) {
          dx = kEpsilon;
        }
        float grad = dy / dx;
        Double theta = Math.atan(grad);
        theta = kNumBins * (0.5 + theta / Math.PI);
        Integer bin = theta.intValue();
        if (bin < 0) {
          bin = 0;
        }
        if (bin >= kNumBins) {
          bin = kNumBins - 1;
        }
        atanTable[x][y] = bin.byteValue();
      }
    }
    return atanTable;
  }

  @Override public String featureName() {
    return "hog";
  }

  // There are 9 orientation bins. This was determined
  // to be the optimal bin size by the Dalal and Triggs paper.
  @Override public int featureSize() {
    return kNumBins;
  }

  @Override
  public void analyze(BufferedImage image) {
    imgRef = image;
    int width = image.getWidth();
    int height = image.getHeight();
    gradientMagnitude = new float[width][height];
    gradientBin = new byte[width][height];
    int[][] lum = new int[width][height];
    // Compute sum of all channels per pixel
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int pixel = image.getRGB(x, y);
        for (int i = 0; i < 3; i++) {
          lum[x][y] += pixel & 0xff;
          pixel = pixel >> 8;
        }
        lum[x][y] /= 3;
      }
    }
    for (int y = 1; y < height - 1; y++) {
      for (int x = 1; x < width - 1; x++) {
        int dx = lum[x + 1][y] - lum[x - 1][y];
        int dy = lum[x][y + 1] - lum[x][y - 1];
        byte bin = atanTable[dx + 255][dy + 255];
        Double mag = Math.sqrt(dx * dx + dy * dy);
        gradientMagnitude[x][y] = mag.floatValue();
        gradientBin[x][y] = bin;
      }
    }
  }

  @Override
  public float[] extractFeature(int sx, int sy, int ex, int ey) {
    float[] feature = new float[featureSize()];
    if (sx >= ex || sy >= ey ||
        ex > imgRef.getWidth() || ey > imgRef.getHeight()) {
      return feature;
    }
    float sum = 0;
    for (int y = sy; y < ey; y++) {
      for (int x = sx; x < ex; x++) {
        for (int i = 0; i < kNumBins; i++) {
          byte bin = gradientBin[x][y];
          float mag = gradientMagnitude[x][y];
          feature[bin] += mag;
          sum += mag;
        }
      }
    }
    if (sum > 0) {
      float scale = 1.0f / sum;
      for (int i = 0; i < featureSize(); i++) {
        feature[i] *= scale;
      }
    }
    return feature;
  }
}
