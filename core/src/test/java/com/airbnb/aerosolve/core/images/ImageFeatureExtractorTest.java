package com.airbnb.aerosolve.core.images;

import com.airbnb.aerosolve.core.features.DenseVector;
import com.airbnb.aerosolve.core.features.FamilyVector;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import java.awt.image.BufferedImage;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

@Slf4j
public class ImageFeatureExtractorTest {
  private final FeatureRegistry registry = new FeatureRegistry();

  public void validateFeature(MultiFamilyVector vector,
                              String name,
                              int expectedCount) {
    assertTrue(vector.contains(name));
    FamilyVector fVec = vector.get(name);
    assertTrue(fVec.size() == expectedCount);
    assertTrue(fVec instanceof DenseVector);
    log.info("feature " + name + "[0] = " + ((DenseVector) fVec).denseArray()[0]);
  }

  public void validateFeatureVector(MultiFamilyVector featureVector) {
    assertTrue(featureVector.numFamilies() == 4);
    assertTrue(featureVector.contains("rgb"));
    final int kNumGrids = 1 + 4 + 16;
    validateFeature(featureVector, "rgb", 512 * kNumGrids);
    validateFeature(featureVector, "hog", 9 * kNumGrids);
    validateFeature(featureVector, "lbp", 256 * kNumGrids);
    validateFeature(featureVector, "hsv", 65 * kNumGrids);
  }

  @Test
  public void testBlackImage() {
    BufferedImage image = new BufferedImage(10, 10, BufferedImage.TYPE_BYTE_GRAY);
    ImageFeatureExtractor featureExtractor = ImageFeatureExtractor.getInstance();
    MultiFamilyVector featureVector = featureExtractor.getFeatureVector(image, registry);
    validateFeatureVector(featureVector);
  }
}
