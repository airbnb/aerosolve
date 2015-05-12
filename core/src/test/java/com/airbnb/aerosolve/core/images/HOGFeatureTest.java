package com.airbnb.aerosolve.core.images;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.util.List;

import static org.junit.Assert.assertTrue;

public class HOGFeatureTest {
  private static final Logger log = LoggerFactory.getLogger(HOGFeatureTest.class);

  // There should be no gradients in a black image. All signals are zero.
  @Test
  public void testBlackImage() {
    BufferedImage image = new BufferedImage(10, 10, BufferedImage.TYPE_BYTE_GRAY);
    for (int x = 0; x < 10; x++) {
      for (int y = 0; y < 10; y++) {
        image.setRGB(x, y, 0);
      }
    }
    ImageFeature hog = new HOGFeature();
    hog.analyze(image);
    float[] feature = hog.extractFeature(0, 0, 10, 10);
    assertTrue(feature.length == hog.featureSize());
    for (int i = 0; i < hog.featureSize(); i++) {
      assertTrue(feature[i] <= 0.1);
    }
  }

  // Only the middle bin where theta is 0 should be active.
  @Test
  public void testVerticalStripe() {
    BufferedImage image = new BufferedImage(10, 10, BufferedImage.TYPE_INT_ARGB);
    for (int x = 0; x < 10; x++) {
      for (int y = 0; y < 10; y++) {
        if (x % 3 == 0) {
          image.setRGB(x, y, 0xffffff);
        } else {
          image.setRGB(x, y, 0);
        }
      }
    }
    ImageFeature hog = new HOGFeature();
    hog.analyze(image);
    float[] feature = hog.extractFeature(0, 0, 10, 10);
    assertTrue(feature.length == hog.featureSize());
    for (int i = 0; i < hog.featureSize(); i++) {
      if (feature[i] > 0.0) {
        log.info("Feature " + i + "=" + feature[i]);
      }
      if (i == hog.featureSize() / 2) {
        assertTrue(feature[i] > 0.9);
        assertTrue(feature[i] < 1.1);
      } else {
        assertTrue(feature[i] >= 0.0);
        assertTrue(feature[i] <= 0.1);
      }
    }
  }

  // Only the first and last bins where theta is -pi/2 or pi/2 should be active.
  @Test
  public void testHorizontalStripe() {
    BufferedImage image = new BufferedImage(10, 10, BufferedImage.TYPE_INT_ARGB);
    for (int x = 0; x < 10; x++) {
      for (int y = 0; y < 10; y++) {
        if (y % 3 == 0) {
          image.setRGB(x, y, 0xaabbcc);
        } else {
          image.setRGB(x, y, 0);
        }
      }
    }
    ImageFeature hog = new HOGFeature();
    hog.analyze(image);
    float[] feature = hog.extractFeature(0, 0, 10, 10);
    assertTrue(feature.length == hog.featureSize());
    for (int i = 0; i < hog.featureSize(); i++) {
      if (feature[i] > 0.0) {
        log.info("Feature " + i + "=" + feature[i]);
      }
      // Horizontal lines can have angles -pi/2 or pi/2
      if (i == 0 || i == hog.featureSize() - 1) {
        assertTrue(feature[i] >= 0.49);
        assertTrue(feature[i] <= 0.51);
      } else {
        assertTrue(feature[i] >= 0.0);
        assertTrue(feature[i] <= 0.1);
      }
    }
  }

  // Draw a circle and see what the histogram of edges looks like.
  // Since all angles are represented it should be more or less uniform.
  @Test
  public void testCircle() {
    final int kSize = 32;
    final int kRadiusSquared = 10 * 10;
    BufferedImage image = new BufferedImage(kSize, kSize, BufferedImage.TYPE_INT_ARGB);
    for (int x = 0; x < kSize; x++) {
      for (int y = 0; y < kSize; y++) {
        int dx = x - kSize / 2;
        int dy = y - kSize / 2;
        int r2 = dx * dx + dy * dy;
        if (r2 <= kRadiusSquared) {
          image.setRGB(x, y, r2);
        } else {
          image.setRGB(x, y, 0);
        }
      }
    }
    ImageFeature hog = new HOGFeature();
    hog.analyze(image);
    float[] feature = hog.extractFeature(0, 0, kSize, kSize);
    assertTrue(feature.length == hog.featureSize());
    for (int i = 0; i < hog.featureSize(); i++) {
      if (feature[i] > 0.0) {
        log.info("Feature " + i + "=" + feature[i]);
      }
      assertTrue(feature[i] >= 0.0);
      assertTrue(feature[i] <= 0.2);
    }
  }

  // The middle bins for each level of the pyramid should be active.
  @Test
  public void testVerticalStripeSPMK() {
    BufferedImage image = new BufferedImage(32, 32, BufferedImage.TYPE_INT_ARGB);
    for (int x = 0; x < 32; x++) {
      for (int y = 0; y < 32; y++) {
        if (x % 3 == 0) {
          image.setRGB(x, y, 0xffffff);
        } else {
          image.setRGB(x, y, 0);
        }
      }
    }
    ImageFeature hog = new HOGFeature();
    List<Float> feature = hog.extractFeatureSPMK(image);
    final int numBlocks = 1 + 2 * 2 + 4 * 4;
    log.info("Feature size with SPMK = " + feature.size());
    assertTrue(feature.size() == hog.featureSize() * numBlocks);
    for (int i = 0; i < hog.featureSize() * numBlocks; i++) {
      if (feature.get(i) > 0.0) {
        log.info("Feature " + i + "=" + feature.get(i));
      }
      if (i % hog.featureSize() == hog.featureSize() / 2) {
        float weight = 0.25f;
        // The first 1x1, 2x2 blocks should all be weight 0.25
        // after that it should be weight 0.5
        if (i > hog.featureSize() * 5) {
          weight = 0.5f;
        }
        assertTrue(feature.get(i) > weight - 0.01);
        assertTrue(feature.get(i) < weight + 0.01);
      } else {
        assertTrue(feature.get(i) >= 0.0);
        assertTrue(feature.get(i) <= 0.1);
      }
    }
  }
}
