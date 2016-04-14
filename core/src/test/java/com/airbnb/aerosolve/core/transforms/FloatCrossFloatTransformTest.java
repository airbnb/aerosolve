package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.google.common.collect.ImmutableMap;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * Created by christhetree on 4/8/16.
 */
public class FloatCrossFloatTransformTest extends BaseTransformTest {

  public String makeConfig() {
    return "test_float_cross_float {\n" +
        " transform : float_cross_float\n" +
        " field1 : floatFeature1\n" +
        " bucket : 1.0\n" +
        " cap : 1000.0\n" +
        " field2 : floatFeature2\n" +
        " output : out\n" +
        "}";
  }

  @Override
  public String configKey() {
    return "test_float_cross_float";
  }

  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .sparse("floatFeature1", "x", 50.0)
        .sparse("floatFeature1", "y", 1.3)
        .sparse("floatFeature1", "z", 2000.0)
        .sparse("floatFeature2", "i", 1.2)
        .sparse("floatFeature2", "j", 3.4)
        .sparse("floatFeature2", "k", 5.6)
        .build();
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);
    assertTrue(featureVector.numFamilies() == 3);

    assertSparseFamily(featureVector, "out", 9, ImmutableMap.<String, Double>builder()
                           .put("x=50.0^i", 1.2)
                           .put("y=1.0^i", 1.2)
                           .put("z=1000.0^i", 1.2)
                           .put("x=50.0^j", 3.4)
                           .put("y=1.0^j", 3.4)
                           .put("z=1000.0^j", 3.4)
                           .put("x=50.0^k", 5.6)
                           .put("y=1.0^k", 5.6)
                           .put("z=1000.0^k", 5.6)
                           .build()
    );
  }
}
