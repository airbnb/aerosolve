package com.airbnb.aerosolve.core.images;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.util.List;

import static org.junit.Assert.assertTrue;

public class LBPFeatureTest {
  private static final Logger log = LoggerFactory.getLogger(LBPFeatureTest.class);

  // There should be no gradients in a black image. All signals are zero.
  @Test
  public void testBlackImage() {
    BufferedImage image = new BufferedImage(10, 10, BufferedImage.TYPE_BYTE_GRAY);
    for (int x = 0; x < 10; x++) {
      for (int y = 0; y < 10; y++) {
        image.setRGB(x, y, 0);
      }
    }
    ImageFeature lbp = new LBPFeature();
    lbp.analyze(image);
    float[] feature = lbp.extractFeature(0, 0, 10, 10);
    assertTrue(feature.length == lbp.featureSize());
    for (int i = 0; i < lbp.featureSize(); i++) {
      if (i == 0) {
        assertTrue(feature[i] >= 0.9);
        assertTrue(feature[i] <= 1.1);
      } else {
        assertTrue(feature[i] >= 0.0);
        assertTrue(feature[i] <= 0.1);
      }
    }
  }

  // Half the pixels have no left or right bar (so no bit set)
  // the rest are all either on the left (bits 1, 8, 32 are set)
  // or right (4, 16, 128)
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
    ImageFeature lbp = new LBPFeature();
    lbp.analyze(image);
    float[] feature = lbp.extractFeature(0, 0, 10, 10);
    assertTrue(feature.length == lbp.featureSize());
    for (int i = 0; i < lbp.featureSize(); i++) {
      if (feature[i] > 0.0) {
        log.info("Feature " + i + "=" + feature[i]);
      }
      if (i == 0) {
        assertTrue(feature[i] > 0.45);
        assertTrue(feature[i] < 0.55);
      } else if (i == 1 + 8 + 32 || i == 4 + 16 + 128){
        assertTrue(feature[i] >= 0.2);
        assertTrue(feature[i] <= 0.3);
      }
    }
  }

  // Half the pixels have no top or bottom bar (so no bit set)
  // the rest are all either on the above (bits 1, 2, 4 are set)
  // or below (32, 64, 128)
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
    ImageFeature lbp = new LBPFeature();
    lbp.analyze(image);
    float[] feature = lbp.extractFeature(0, 0, 10, 10);
    assertTrue(feature.length == lbp.featureSize());
    for (int i = 0; i < lbp.featureSize(); i++) {
      if (feature[i] > 0.0) {
        log.info("Feature " + i + "=" + feature[i]);
      }
      if (i == 0) {
        assertTrue(feature[i] > 0.45);
        assertTrue(feature[i] < 0.55);
      } else if (i == 1 + 2 + 4 || i == 32 + 64 + 128){
        assertTrue(feature[i] >= 0.2);
        assertTrue(feature[i] <= 0.3);
      }
    }
  }
}
