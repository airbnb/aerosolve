package com.airbnb.aerosolve.core.images;

import java.awt.image.BufferedImage;
import java.awt.Color;
import java.lang.Override;
import java.lang.Math;

/*
 Creates a histogram of Hue, saturation, value.
 This is a different color space than rgb and focuses more on color.
  */

public class HSVFeature extends ImageFeature {
  private int[][] hsv = null;
  private BufferedImage imgRef = null;
  // Number of buckets for hue and saturation.
  private final int kSatBuckets = 4;
  private final int kHueBuckets = 16;
  // The cutoff for brightness.
  private static float kMinBrightness = 0.25f;

  @Override public String featureName() {
    return "hsv";
  }

  // There are kBuckets ^ 2 bins for hue and saturation and one more for "dark".
  @Override public int featureSize() {
    return 1 + kSatBuckets * kHueBuckets;
  }

  @Override
  public void analyze(BufferedImage image) {
    imgRef = image;
    int width = image.getWidth();
    int height = image.getHeight();
    hsv = new int[width][height];
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int pixel = image.getRGB(x, y);
        int r = (pixel >> 16) & 0xff;
        int g = (pixel >> 8) & 0xff;
        int b = pixel & 0xff;
        float[] hsb = Color.RGBtoHSB(r, g, b, null);
        int token = 0; // Dark.
        // We really only care about the color so discard any
        // dark color less than 25% brightness.
        if (hsb[2] > kMinBrightness) {
          int hue = Math.round((kHueBuckets - 1) * hsb[0]);
          int sat = Math.round((kSatBuckets - 1) * hsb[1]);
          token = 1 + kHueBuckets * sat + hue;
        }
        hsv[x][y] = token;
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
        feature[hsv[x][y]]++;
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
