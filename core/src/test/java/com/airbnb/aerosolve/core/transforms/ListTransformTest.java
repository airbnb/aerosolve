package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class ListTransformTest extends BaseTransformTest {

  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.makeSimpleVector(registry);
  }

  public String makeConfig() {
    return "test_quantize {\n" +
           " transform : quantize\n" +
           " field1 : loc\n" +
           " scale : 10\n" +
           " output : loc_quantized\n" +
           "}\n" +
           "test_cross {\n" +
           " transform : cross\n" +
           " field1 : strFeature1\n" +
           " field2 : loc_quantized\n" +
           " output : out\n" +
           "}\n" +
           "test_list {\n" +
           " transform : list\n" +
           " transforms : [test_quantize, test_cross]\n" +
           "}";
  }

  @Override
  public String configKey() {
    return "test_list";
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 4);

    assertStringFamily(featureVector, "out", 4,
                       ImmutableSet.of("bbb^long=400",
                                       "aaa^lat=377"));
  }
}