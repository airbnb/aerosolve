package com.airbnb.aerosolve.core.images;

import com.airbnb.aerosolve.core.FeatureVector;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.util.Map;
import java.util.List;

import static org.junit.Assert.assertTrue;

public class ImageFeatureExtractorTest {
  private static final Logger log = LoggerFactory.getLogger(ImageFeatureExtractorTest.class);

  public void validateFeature(Map<String, List<Double>> denseFeatures,
                              String name,
                              int expectedCount) {
    assertTrue(denseFeatures.containsKey(name));
    assertTrue(denseFeatures.get(name).size() == expectedCount);
    log.info("feature " + name + "[0] = " + denseFeatures.get(name).get(0));
  }

  public void validateFeatureVector(FeatureVector featureVector) {
    Map<String, List<Double>> denseFeatures = featureVector.getDenseFeatures();
    assertTrue(denseFeatures != null);
    assertTrue(denseFeatures.containsKey("rgb"));
    final int kNumGrids = 1 + 4 + 16;
    validateFeature(denseFeatures, "rgb", 512 * kNumGrids);
    validateFeature(denseFeatures, "hog", 9 * kNumGrids);
    validateFeature(denseFeatures, "lbp", 256 * kNumGrids);
    validateFeature(denseFeatures, "hsv", 65 * kNumGrids);
  }

  @Test
  public void testBlackImage() {
    BufferedImage image = new BufferedImage(10, 10, BufferedImage.TYPE_BYTE_GRAY);
    ImageFeatureExtractor featureExtractor = ImageFeatureExtractor.getInstance();
    FeatureVector featureVector = featureExtractor.getFeatureVector(image);
    validateFeatureVector(featureVector);
  }
}
