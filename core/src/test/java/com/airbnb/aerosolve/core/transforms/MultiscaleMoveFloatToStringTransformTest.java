package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class MultiscaleMoveFloatToStringTransformTest extends BaseTransformTest {

  public String makeConfig() {
    return "test_quantize {\n" +
           " transform : multiscale_move_float_to_string\n" +
           " field1 : loc\n" +
           " buckets : [1.0, 10.0]\n" +
           " keys : [lat]\n" +
           " output : loc_quantized\n" +
           "}";
  }

  @Override
  public String configKey() {
    return "test_quantize";
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = TransformTestingHelper.makeSimpleVector(registry);
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 3);

    assertStringFamily(featureVector, "loc_quantized", 2,
                       ImmutableSet.of("lat[1.0]=37.0", "lat[10.0]=30.0"));

  }
}