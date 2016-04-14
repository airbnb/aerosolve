package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableMap;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class BucketTransformTest extends BaseTransformTest {

  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .simpleStrings()
        .sparse("loc", "lat", 37.7)
        .sparse("loc", "long", 40.4)
        .sparse("loc", "zero", 0.0)
        .sparse("loc", "negative", -1.5)
        .build();
  }

  public String makeConfig() {
    return "test_quantize {\n" +
           " transform : bucket_float\n" +
           " field1 : loc\n" +
           " bucket : 1.0\n" +
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
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 3);

    assertSparseFamily(featureVector, "loc_quantized", 4,
                       ImmutableMap.of("lat[1.0]=37.0", 0.7,
                                       "long[1.0]=40.0", 0.4,
                                       "zero[1.0]=0.0", 0.0,
                                       "negative[1.0]=-1.0", -0.5));
  }
}