package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class DeleteFeatureTransformTest extends BaseTransformTest {

  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.makeSimpleVector(registry);
  }

  public String makeConfig() {
    return "test_delete {\n" +
           " transform : delete_float_feature\n" +
           " field1 : loc\n" +
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

    assertSparseFamily(featureVector, "loc", 1,
                       ImmutableMap.of("lat", 37.7),
                       ImmutableSet.of("long"));

    assertStringFamily(featureVector, "strFeature1", 2,
                       ImmutableSet.of("aaa", "bbb"));
  }
}