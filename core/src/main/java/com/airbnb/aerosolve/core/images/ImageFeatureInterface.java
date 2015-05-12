package com.airbnb.aerosolve.core.images;

import java.awt.image.BufferedImage;

interface ImageFeatureInterface {
  // Returns the size of the feature.
  int featureSize();
  // Pre-computes features in the image
  void analyze(BufferedImage image);
  // Extracts features from an image inside the window
  // [sx, sy] to (ex, ey)
  float[] extractFeature(int sx, int sy, int ex, int ey);
}
