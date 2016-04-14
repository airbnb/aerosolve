package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableMap;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class NormalizeFloatTransformTest extends BaseTransformTest {
  
  public String makeConfig() {
    return "test_norm {\n" +
           " transform : normalize_float\n" +
           " field1 : loc\n" +
           "}";
  }

  @Override
  public String configKey() {
    return "test_norm";
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = TransformTestingHelper.makeFoobarVector(registry);
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 4);

    double scale = 1.0 / Math.sqrt(37.7 * 37.7 + 40.0 * 40.0 + 20.0 * 20.0);

    assertSparseFamily(featureVector, "loc", 3,
                       ImmutableMap.of("lat", scale * 37.7,
                                       "long", scale * 40.0,
                                       "z", scale * -20.0));
  }

}