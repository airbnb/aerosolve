package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableMap;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class ProductTransformTest extends BaseTransformTest {

  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .location()
        .sparse("loc", "foo", 7.0)
        .build();
  }

  public String makeConfig() {
    return "test_prod {\n" +
           " transform : product\n" +
           " field1 : loc\n" +
           " keys: [lat,long]\n" +
           " output : loc_prod\n" +
           "}";
  }

  @Override
  public String configKey() {
    return "test_prod";
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 2);

    assertSparseFamily(featureVector, "loc_prod", 1,
                       ImmutableMap.of("*", (1 + 37.7) * (1 + 40.0)));
  }
}