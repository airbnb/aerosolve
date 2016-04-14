package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class NearestTransformTest extends BaseTransformTest {

  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .simpleStrings()
        .location()
        .sparse("f2", "foo", 41.0)
        .build();
  }
  public String makeConfig() {
    return "test_nearest {\n" +
           " transform : nearest\n" +
           " field1 : loc\n" +
           " field2 : f2\n" +
           " key: foo\n" +
           " output : nearest\n" +
           "}";
  }

  @Override
  public String configKey() {
    return "test_nearest";
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 4);

    assertStringFamily(featureVector, "nearest", 1,
                       ImmutableSet.of("foo~=long"));
  }
}