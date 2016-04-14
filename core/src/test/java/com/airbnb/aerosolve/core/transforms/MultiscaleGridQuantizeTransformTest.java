package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
@Slf4j
public class MultiscaleGridQuantizeTransformTest extends BaseTransformTest {

  public String makeConfig() {
    return "test_quantize {\n" +
           " transform : multiscale_grid_quantize\n" +
           " field1 : loc\n" +
           " value1 : lat\n" +
           " value2 : long\n" +
           " buckets : [1, 10]\n" +
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
                       ImmutableSet.of("[10.0]=(30.0,40.0)", "[1.0]=(37.0,40.0)"));
  }
}