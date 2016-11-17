package com.airbnb.aerosolve.core.images;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

public abstract class ImageFeature implements ImageFeatureInterface {
  public abstract String featureName();
  public abstract int featureSize();
  public abstract void analyze(BufferedImage image);
  public abstract float[] extractFeature(int sx, int sy, int ex, int ey);
  /* The following functions compute the Spatial Pyramid Match Kernel
  for the feature. This is a weighted sum of different partitions of an image.
  The weights for levels 0, 1, 2 are [0.25, 0.25, 0.5] respectively
  so higher weights are given to finer subdivisions following the
  formula given in the paper.
  See http://www-cvr.ai.uiuc.edu/ponce_grp/publication/paper/cvpr06b.pdf
   */
  public void addAndScale(List<Float> featureList,
                          float[] featureInput, int count, float scale) {
    for (int i = 0; i < count; i++) {
      featureList.add(scale * featureInput[i]);
    }
  }

  public void SPMKHelper(List<Float> featureList,
                         int sx, int sy, int ex, int ey,
                         int splits, float scale) {
    if (splits <= 1) {
      float tmp[] = extractFeature(sx, sy, ex, ey);
      addAndScale(featureList, tmp, featureSize(), scale);
      return;
    }
    int dx = (ex - sx) / splits;
    int dy = (ey - sy) / splits;
    for (int i = 0; i < splits; i++) {
      for (int j = 0; j < splits; j++) {
        int x = sx + i * dx;
        int y = sy + j * dy;
        float tmp[] = extractFeature(x, y, x + dx, y + dy);
        addAndScale(featureList, tmp, featureSize(), scale);
      }
    }
  }

  public List<Float> extractFeatureSPMK(BufferedImage image) {
    List<Float> feature = new ArrayList<>();
    final int kPad = 3;
    int width = image.getWidth() - kPad;
    int height = image.getHeight() - kPad;

    analyze(image);
    SPMKHelper(feature, kPad, kPad, width, height, 1, 0.25f);
    SPMKHelper(feature, kPad, kPad, width, height, 2, 0.25f);
    SPMKHelper(feature, kPad, kPad, width, height, 4, 0.5f);
    return feature;
  }
}