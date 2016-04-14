package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class SelfCrossTransformTest extends BaseTransformTest {

  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .simpleStrings()
        .build();
  }

  public String makeConfig() {
    return "test_cross {\n" +
           " transform : self_cross\n" +
           " field1 : strFeature1\n" +
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
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 2);

    assertStringFamily(featureVector, "out", 1, ImmutableSet.of(
        "aaa^bbb"
    ));
  }
}