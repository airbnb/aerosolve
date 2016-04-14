package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class QuantizeTransformTest extends BaseTransformTest {

  public String makeConfig() {
    return "test_quantize {\n" +
           " transform : quantize\n" +
           " field1 : loc\n" +
           " scale : 10\n" +
           " output : loc_quantized\n" +
           "}";
  }

  @Override
  public String configKey() {
    return "test_quantize";
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = TransformTestingHelper.makeSimpleVector(registry);
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 3);

    assertStringFamily(featureVector, "loc_quantized", 2,
                       ImmutableSet.of("lat=377", "long=400"));
  }
}