package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class DeleteFloatFeatureFamilyTransformTest extends BaseTransformTest {

  public String makeConfig() {
    return "test_delete_float_feature_family {\n" +
        " transform: delete_float_feature_family\n" +
        " fields: [F1, F2, F3]" +
        "}";
  }

  @Override
  public String configKey() {
    return "test_delete_float_feature_family";
  }

  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .sparse("F1", "A", 1.0)
        .sparse("F1", "B", 2.0)
        .sparse("F2", "C", 3.0)
        .sparse("F2", "D", 4.0)
        .sparse("F3", "E", 5.0)
        .sparse("F3", "F", 6.0)
        .sparse("F4", "G", 7.0)
        .sparse("F4", "H", 8.0)
        .build();
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = makeFeatureVector();

    assertTrue(featureVector.contains(registry.family("F1")));
    assertTrue(featureVector.contains(registry.family("F2")));
    assertTrue(featureVector.contains(registry.family("F3")));
    assertTrue(featureVector.contains(registry.family("F4")));
    assertEquals(4, featureVector.numFamilies());

    transform.apply(featureVector);
    assertFalse(featureVector.contains(registry.family("F1")));
    assertFalse(featureVector.contains(registry.family("F2")));
    assertFalse(featureVector.contains(registry.family("F3")));
    assertTrue(featureVector.contains(registry.family("F4")));
    assertEquals(1, featureVector.numFamilies());
  }
}
