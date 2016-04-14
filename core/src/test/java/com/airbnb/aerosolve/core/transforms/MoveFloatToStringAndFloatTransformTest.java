package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * Created by christhetree on 4/8/16.
 */
@Slf4j
public class MoveFloatToStringAndFloatTransformTest extends BaseTransformTest {

  public String makeConfig() {
    return "test_move_float_to_string_and_float {\n" +
        " transform : move_float_to_string_and_float\n" +
        " field1 : floatFeature1\n" +
        " keys : [a, b, c, d, e, f, g, h, i]\n" +
        " bucket : 1.0\n" +
        " max_bucket : 10.0\n" +
        " min_bucket : 0.0\n" +
        " string_output : stringOutput\n" +
        " float_output : floatOutput\n" +
        "}";
  }

  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .simpleStrings()
        .sparse("floatFeature1", "a", 0.0)
        .sparse("floatFeature1", "b", 10.0)
        .sparse("floatFeature1", "c", 9.9)
        .sparse("floatFeature1", "d", 10.1)
        .sparse("floatFeature1", "e", 11.01)
        .sparse("floatFeature1", "f", -0.1)
        .sparse("floatFeature1", "g", -1.01)
        .sparse("floatFeature1", "h", 21.3)
        .sparse("floatFeature1", "i", 2000.0)
        .sparse("floatFeature1", "j", 1.0)
        .sparse("floatFeature1", "k", 9000.0)
        .build();
  }

  @Override
  public String configKey() {
    return "test_move_float_to_string_and_float";
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    // TODO (Brad): This test is failing because I intentionally output to a single family
    // pending a conversation with Chris about this new transform.
    assertTrue(featureVector.numFamilies() == 2);

    assertStringFamily(featureVector, "stringOutput", 5,
                       ImmutableSet.of("a=0.0",
                                       "b=10.0",
                                       "c=9.0",
                                       "d=10.0",
                                       "f=0.0"));

    assertSparseFamily(featureVector, "floatOutput", 4,
                       ImmutableMap.of("e", 11.01,
                                       "g", -1.01,
                                       "h", 21.3,
                                       "i", 2000.0));
  }
}
