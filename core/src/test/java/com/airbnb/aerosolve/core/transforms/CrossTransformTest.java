package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class CrossTransformTest extends BaseTransformTest {

  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .string("feature1", "aaa")
        .string("feature1", "bbb")
        .string("feature2", "11")
        .string("feature2", "22")
        .build();
  }

  public String makeConfig() {
    return "test_cross {\n" +
           " transform : cross\n" +
           " field1 : feature1\n" +
           " field2 : feature2\n" +
           " output : out\n" +
           "}";
  }

  @Override
  public String configKey() {
    return "test_cross";
  }

  @Test
  public void testTransform() {
    MultiFamilyVector featureVector = makeFeatureVector();
    transformVector(featureVector);
    assertTrue(featureVector.numFamilies() == 3);

    assertStringFamily(featureVector, "out", 4, ImmutableSet.of(
        "aaa^11",
        "aaa^22",
        "bbb^11",
        "bbb^22"
    ));
  }
}