package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableMap;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * Created by seckcoder on 12/17/15.
 */
public class DateDiffTransformTest extends BaseTransformTest {
  public String makeConfig() {
    return "test_datediff {\n" +
            " transform: date_diff\n" +
            " field1: endDates\n" +
            " field2: startDates\n" +
            " output: bar\n" +
            "}";
  }

  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .string("endDates", "2009-03-01")
        .string("startDates", "2009-02-27")
        .build();
  }

  @Override
  public String configKey() {
    return "test_datediff";
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);
    assertTrue(featureVector.numFamilies() == 3);

    assertSparseFamily(featureVector, "bar", 1, ImmutableMap.of(
        "2009-03-01-m-2009-02-27", 2.0
    ));
  }
}
