package com.airbnb.aerosolve.core.images;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.util.List;

import static org.junit.Assert.assertTrue;

public class RGBFeatureTest {
  private static final Logger log = LoggerFactory.getLogger(RGBFeatureTest.class);

  @Test
  public void testBlackImage() {
    BufferedImage image = new BufferedImage(10, 10, BufferedImage.TYPE_BYTE_GRAY);
    for (int x = 0; x < 10; x++) {
      for (int y = 0; y < 10; y++) {
        image.setRGB(x, y, 0);
      }
    }
    ImageFeature rgb = new RGBFeature();
    rgb.analyze(image);
    float[] feature = rgb.extractFeature(0, 0, 10, 10);
    assertTrue(feature.length == rgb.featureSize());
    for (int i = 0; i < rgb.featureSize(); i++) {
      if (feature[i] > 0.0) {
        log.info("Feature " + i + "=" + feature[i]);
      }
      if (i == 0) {
        assertTrue(feature[i] > 0.9);
        assertTrue(feature[i] < 1.1);
      } else {
        assertTrue(feature[i] >= 0.0);
        assertTrue(feature[i] <= 0.1);
      }
    }
  }

  @Test
  public void testWhiteImage() {
    BufferedImage image = new BufferedImage(10, 10, BufferedImage.TYPE_BYTE_GRAY);
    for (int x = 0; x < 10; x++) {
      for (int y = 0; y < 10; y++) {
        image.setRGB(x, y, 0xffffff);
      }
    }
    ImageFeature rgb = new RGBFeature();
    rgb.analyze(image);
    float[] feature = rgb.extractFeature(0, 0, 10, 10);
    assertTrue(feature.length == rgb.featureSize());
    for (int i = 0; i < rgb.featureSize(); i++) {
      if (feature[i] > 0.0) {
        log.info("Feature " + i + "=" + feature[i]);
      }
      if (i == rgb.featureSize() - 1) {
        assertTrue(feature[i] > 0.9);
        assertTrue(feature[i] < 1.1);
      } else {
        assertTrue(feature[i] >= 0.0);
        assertTrue(feature[i] <= 0.1);
      }
    }
  }
  
  @Test
  public void testColorImage() {
    BufferedImage image = new BufferedImage(10, 10, BufferedImage.TYPE_INT_ARGB);
    for (int x = 0; x < 10; x++) {
      for (int y = 0; y < 10; y++) {
        image.setRGB(x, y, 0xa3b2c1);
      }
    }
    int token = (0xa >> 1 & 0x7) << 6 |
                (0xb >> 1 & 0x7) << 3 |
                (0xc >> 1 & 0x7) << 0;
    ImageFeature rgb = new RGBFeature();
    rgb.analyze(image);
    float[] feature = rgb.extractFeature(0, 0, 10, 10);
    assertTrue(feature.length == rgb.featureSize());
    for (int i = 0; i < rgb.featureSize(); i++) {
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
  public void testColorImageSPMK() {
    BufferedImage image = new BufferedImage(10, 10, BufferedImage.TYPE_INT_ARGB);
    for (int x = 0; x < 10; x++) {
      for (int y = 0; y < 10; y++) {
        image.setRGB(x, y, 0xa3b2c1);
      }
    }
    int token = (0xa >> 1 & 0x7) << 6 |
                (0xb >> 1 & 0x7) << 3 |
                (0xc >> 1 & 0x7) << 0;
    ImageFeature rgb = new RGBFeature();
    List<Float> feature = rgb.extractFeatureSPMK(image);
    final int numBlocks = 1 + 2 * 2 + 4 * 4;
    log.info("Feature size with SPMK = " + feature.size());
    assertTrue(feature.size() == rgb.featureSize() * numBlocks);
    for (int i = 0; i < rgb.featureSize() * numBlocks; i++) {
      if (feature.get(i) > 0.0) {
        log.info("Feature " + i + "=" + feature.get(i));
      }
      if (i % rgb.featureSize() == token) {
        float weight = 0.25f;
        // The first 1x1, 2x2 blocks should all be weight 0.25
        // after that it should be weight 0.5
        if (i > rgb.featureSize() * 5) {
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
  
  @Test
  public void testTwoColorImage() {
    BufferedImage image = new BufferedImage(10, 10, BufferedImage.TYPE_INT_ARGB);
    for (int x = 0; x < 10; x++) {
      for (int y = 0; y < 10; y++) {
        if (x % 2 == 0) {
          image.setRGB(x, y, 0xa3b2c1);
        } else {
          image.setRGB(x, y, 0xd5e6f7);
        }
      }
    }
    int token1 = (0xa >> 1 & 0x7) << 6 |
                 (0xb >> 1 & 0x7) << 3 |
                 (0xc >> 1 & 0x7) << 0;
    int token2 = (0xd >> 1 & 0x7) << 6 |
                 (0xe >> 1 & 0x7) << 3 |
                 (0xf >> 1 & 0x7) << 0;
    ImageFeature rgb = new RGBFeature();
    rgb.analyze(image);
    float[] feature = rgb.extractFeature(0, 0, 10, 10);
    assertTrue(feature.length == rgb.featureSize());
    for (int i = 0; i < rgb.featureSize(); i++) {
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
