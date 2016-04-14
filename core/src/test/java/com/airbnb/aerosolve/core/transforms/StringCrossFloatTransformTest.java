package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableMap;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class StringCrossFloatTransformTest extends BaseTransformTest {

  public String makeConfig() {
    return "test_cross {\n" +
           " transform : string_cross_float\n" +
           " field1 : strFeature1\n" +
           " field2 : loc\n" +
           " output : out\n" +
           "}";
  }

  @Override
  public String configKey() {
    return "test_cross";
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = TransformTestingHelper.makeSimpleVector(registry);
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 3);

    assertSparseFamily(featureVector, "out", 4, ImmutableMap.of(
        "aaa^lat", 37.7,
        "bbb^lat", 37.7,
        "aaa^long", 40.0,
        "bbb^long", 40.0));
  }
}