package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class DeleteStringFeatureTransformTest extends BaseTransformTest {

  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .simpleStrings()
        .location()
        .string("strFeature1", "aaa:bbbb")
        .build();
  }

  public String makeConfig() {
    return "test_delete {\n" +
           " transform : delete_string_feature\n" +
           " field1 : strFeature1\n" +
           " keys : [long,aaa]\n" +
           "}";
  }

  @Override
  public String configKey() {
    return "test_delete";
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 2);

    assertStringFamily(featureVector, "strFeature1", 1,
                       ImmutableSet.of("bbb"),
                       ImmutableSet.of("aaa", "aaa:bbbb"));
  }
}