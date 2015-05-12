package com.airbnb.aerosolve.core.images;

import java.awt.image.BufferedImage;
import java.lang.Override;

/*
 Creates histograms of quantized RGB images.
 Each pixel r,g,b is quantized into buckets of
 8 colors each. In effect we take the high order
 3 bits of each component and concatenate them
 giving 512 total features.
  */

public class RGBFeature extends ImageFeature {
  private int[][] rgb = null;
  private BufferedImage imgRef = null;

  @Override public String featureName() {
    return "rgb";
  }

  // There are three rgb components each with
  // 8 values so total 8 ^ 3 variations.
  @Override public int featureSize() {
    return 512;
  }

  @Override
  public void analyze(BufferedImage image) {
    imgRef = image;
    int width = image.getWidth();
    int height = image.getHeight();
    rgb = new int[width][height];
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int pixel = image.getRGB(x, y);
        int token = 0;
        for (int i = 0; i < 3; i++) {
          // Mask out the high order 3 bits and shift
          // right 5-bits to get the high order component.
          // Left shift by the proper 3 bits to put them in place.
          token |= ((pixel >> 5) & 0x7) << i * 3;
          pixel = pixel >> 8;
        }
        rgb[x][y] = token;
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
    int count = 0;
    for (int y = sy; y < ey; y++) {
      for (int x = sx; x < ex; x++) {
        count++;
        feature[rgb[x][y]]++;
      }
    }
    if (count > 0) {
      float scale = 1.0f / count;
      for (int i = 0; i < featureSize(); i++) {
        feature[i] *= scale;
      }
    }
    return feature;
  }
}
