package com.airbnb.aerosolve.core.images;

import java.awt.image.BufferedImage;
import java.lang.Override;

/*
 Creates a histogram of local binary patterns
 http://www.mediateam.oulu.fi/publications/pdf/94.p
 http://en.wikipedia.org/wiki/Local_binary_patterns
 */
public class LBPFeature extends ImageFeature {
  private static final int kNumBins = 256;
  private byte[][] lbpBin = null;
  private BufferedImage imgRef = null;

  @Override public String featureName() {
    return "lbp";
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
    int[][] lum = new int[width][height];
    // Compute sum of all channels per pixel
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int pixel = image.getRGB(x, y);
        for (int i = 0; i < 3; i++) {
          lum[x][y] += pixel & 0xff;
          pixel = pixel >> 8;
        }
        // Average then divide by 4. The /4 is so that
        // we ignore smaller luminance changes of 4 levels or less.
        lum[x][y] /= 3 * 4;
      }
    }
    lbpBin = new byte[width][height];
    for (int y = 1; y < height - 1; y++) {
      for (int x = 1; x < width - 1; x++) {
        lbpBin[x][y] |= (lum[x - 1][y - 1] > lum[x][y] ? 1 : 0) << 7;
        lbpBin[x][y] |= (lum[x + 0][y - 1] > lum[x][y] ? 1 : 0) << 6;
        lbpBin[x][y] |= (lum[x + 1][y - 1] > lum[x][y] ? 1 : 0) << 5;
        lbpBin[x][y] |= (lum[x - 1][y + 0] > lum[x][y] ? 1 : 0) << 4;
        lbpBin[x][y] |= (lum[x + 1][y + 0] > lum[x][y] ? 1 : 0) << 3;
        lbpBin[x][y] |= (lum[x - 1][y + 1] > lum[x][y] ? 1 : 0) << 2;
        lbpBin[x][y] |= (lum[x + 0][y + 1] > lum[x][y] ? 1 : 0) << 1;
        lbpBin[x][y] |= (lum[x + 1][y + 1] > lum[x][y] ? 1 : 0);
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
          short bin = lbpBin[x][y];
          if (bin < 0) {
            // Since java bytes are signed, fix the negative
            // bytes by adding 128 twice.
            bin += 256;
          }
          feature[bin] += 1.0;
          sum += 1.0;
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
