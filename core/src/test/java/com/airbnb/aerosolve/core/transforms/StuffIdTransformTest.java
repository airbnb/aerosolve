package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableMap;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */

public class StuffIdTransformTest extends BaseTransformTest {


  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .simpleStrings()
        .sparse("FEAT", "searches", 37.7)
        .sparse("ID", "id", 123456789.0)
        .build();
  }

  public String makeConfig() {
    return "test_stuff {\n" +
           " transform : stuff_id\n" +
           " field1 : ID\n" +
           " key1 : id\n" +
           " field2 : FEAT\n" +
           " key2 : searches\n" +
           " output : bar\n" +
           "}";
  }

  @Override
  public String configKey() {
    return "test_stuff";
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 4);

    assertSparseFamily(featureVector, "bar", 1,
                       ImmutableMap.of("searches@123456789", 37.7));
  }
}