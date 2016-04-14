package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.TransformTestingHelper;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * @author Hector Yee
 */
@Slf4j
public class FeatureVectorUtilTest {
  private final FeatureRegistry registry = new FeatureRegistry();

  public MultiFamilyVector makeFeatureVector(String key, double v1, double v2) {
    MultiFamilyVector vector = TransformTestingHelper.makeEmptyVector(registry);
    vector.putDense(registry.family(key), new double[]{v1, v2});
    return vector;
  }

  @Test
  public void testEmptyFeatureVectorMinKernel() {
    FeatureVector a = TransformTestingHelper.makeEmptyVector(registry);
    FeatureVector b = TransformTestingHelper.makeEmptyVector(registry);
    assertEquals(0.0, FeatureVectorUtil.featureVectorMinKernel(a, b), 0.1);
  }

  @Test
  public void testFeatureVectorMinKernelDifferentName() {
    FeatureVector a = makeFeatureVector("a", 1.0, 0.0);
    FeatureVector b = makeFeatureVector("b", 1.0, 0.0);
    assertEquals(0.0, FeatureVectorUtil.featureVectorMinKernel(a, b), 0.1);
  }

  @Test
  public void testFeatureVectorMinKernelNoOverlap() {
    FeatureVector a = makeFeatureVector("a", 1.0, 0.0);
    FeatureVector b = makeFeatureVector("a", 0.0, 1.0);
    assertEquals(0.0, FeatureVectorUtil.featureVectorMinKernel(a, b), 0.1);
  }

  @Test
  public void testFeatureVectorMinKernelSomeOverlap() {
    FeatureVector a = makeFeatureVector("a", 0.3, 0.7);
    FeatureVector b = makeFeatureVector("a", 0.0, 1.0);
    assertEquals(0.7, FeatureVectorUtil.featureVectorMinKernel(a, b), 0.1);
  }
}
