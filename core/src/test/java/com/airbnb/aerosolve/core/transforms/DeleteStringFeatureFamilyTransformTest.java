package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Created by christhetree on 1/29/16.
 */
public class DeleteStringFeatureFamilyTransformTest extends BaseTransformTest {

  public String makeConfig() {
    return "test_delete_string_feature_family {\n" +
        " transform: delete_string_feature_family\n" +
        " fields: [strFeature1, strFeature2, strFeature3]\n" +
        "}";
  }

  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .string("strFeature1", "I am a string in string feature 1")
        .string("strFeature2", "I am a string in string feature 2")
        .string("strFeature3", "I am a string in string feature 3")
        .string("strFeature4", "I am a string in string feature 4")
        .build();
  }

  @Override
  public String configKey() {
    return "test_delete_string_feature_family";
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = makeFeatureVector();

    assertTrue(featureVector.contains(registry.family("strFeature1")));
    assertTrue(featureVector.contains(registry.family("strFeature2")));
    assertTrue(featureVector.contains(registry.family("strFeature3")));
    assertTrue(featureVector.contains(registry.family("strFeature4")));
    assertEquals(4, featureVector.numFamilies());

    transform.apply(featureVector);

    assertFalse(featureVector.contains(registry.family("strFeature1")));
    assertFalse(featureVector.contains(registry.family("strFeature2")));
    assertFalse(featureVector.contains(registry.family("strFeature3")));
    assertTrue(featureVector.contains(registry.family("strFeature4")));
    assertEquals(1, featureVector.numFamilies());
  }
}
