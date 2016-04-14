package com.airbnb.aerosolve.core.images;

import com.airbnb.aerosolve.core.perf.DenseVector;
import com.airbnb.aerosolve.core.perf.Family;
import com.airbnb.aerosolve.core.perf.FamilyVector;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import java.awt.image.BufferedImage;
import java.util.Set;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

@Slf4j
public class ImageFeatureExtractorTest {
  private final FeatureRegistry registry = new FeatureRegistry();

  public void validateFeature(MultiFamilyVector vector,
                              String name,
                              int expectedCount) {
    Family family = registry.family(name);
    assertTrue(vector.contains(family));
    FamilyVector fVec = vector.get(family);
    assertTrue(fVec.size() == expectedCount);
    assertTrue(fVec instanceof DenseVector);
    log.info("feature " + name + "[0] = " + ((DenseVector) fVec).getValues()[0]);
  }

  public void validateFeatureVector(MultiFamilyVector featureVector) {
    assertTrue(featureVector.numFamilies() == 4);
    assertTrue(featureVector.contains(registry.family("rgb")));
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
