package com.airbnb.aerosolve.core.images;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.util.List;

import static org.junit.Assert.assertTrue;

public class HSVFeatureTest {
  private static final Logger log = LoggerFactory.getLogger(HSVFeatureTest.class);

  public void testColorImage(int color, int token) {
    BufferedImage image = new BufferedImage(10, 10, BufferedImage.TYPE_INT_RGB);
    for (int x = 0; x < 10; x++) {
      for (int y = 0; y < 10; y++) {
        image.setRGB(x, y, color);
      }
    }
    ImageFeature hsv = new HSVFeature();
    hsv.analyze(image);
    float[] feature = hsv.extractFeature(0, 0, 10, 10);
    assertTrue(feature.length == hsv.featureSize());
    for (int i = 0; i < hsv.featureSize(); i++) {
      if (feature[i] > 0.0) {
        log.info("Feature " + i + "=" + feature[i]);
      }
      if (i == token) {
        assertTrue(feature[i] > 0.9);
        assertTrue(feature[i] < 1.1);
      } else {
        assertTrue(feature[i] >= 0.0);
        assertTrue(feature[i] <= 0.1);
      }
    }
  }

  @Test
  public void testBlackImage() {
    testColorImage(0, 0);
  }

  @Test
  public void testRedImage() {
    // Saturation should be 3 and hue should be 0.
    testColorImage(0xff0000, 1 + 3 * 16 + 0);
  }

  @Test
  public void testBlueImage() {
    // Saturation should be 3 and hue should be 10.
    // Blue is 240 hue degrees and so 260 / 360 * 16 ~ 10
    testColorImage(0xff, 1 + 3 * 16 + 10);
  }

  @Test
  public void testGreenImage() {
    // Saturation should be 3 and hue should be 5.
    // Green is 120 hue degrees and so 120 / 360 * 16 ~ 5
    testColorImage(0x8200, 1 + 3 * 16 + 5);
  }

  @Test
  public void testYellowImage() {
    // Saturation should be 3 and hue should be 3.
    // Yellow is 60 hue degrees and so 60 / 360 * 16 ~ 3
    testColorImage(0x414100, 1 + 3 * 16 + 3);
  }

  @Test
  public void testWhiteImage() {
    // Saturation should be 0 and hue should be 0.
    testColorImage(0xffffff, 1);
  }

  @Test
  public void testRedGreenImage() {
    BufferedImage image = new BufferedImage(10, 10, BufferedImage.TYPE_INT_RGB);
    for (int x = 0; x < 10; x++) {
      for (int y = 0; y < 10; y++) {
        if (x % 2 == 0) {
          image.setRGB(x, y, 0xff0000);
        } else {
          image.setRGB(x, y, 0x00ff00);
        }
      }
    }
    int token1 = 49;
    int token2 = 54;
    ImageFeature hsv = new HSVFeature();
    hsv.analyze(image);
    float[] feature = hsv.extractFeature(0, 0, 10, 10);
    assertTrue(feature.length == hsv.featureSize());
    for (int i = 0; i < hsv.featureSize(); i++) {
      if (feature[i] > 0.0) {
        log.info("Feature " + i + "=" + feature[i]);
      }
      if (i == token1 || i == token2) {
        assertTrue(feature[i] > 0.49);
        assertTrue(feature[i] < 0.51);
      } else {
        assertTrue(feature[i] >= 0.0);
        assertTrue(feature[i] <= 0.1);
      }
    }
  }
}
