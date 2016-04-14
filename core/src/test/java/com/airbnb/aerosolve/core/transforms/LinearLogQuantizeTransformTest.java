package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class LinearLogQuantizeTransformTest extends BaseTransformTest {

  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .simpleStrings()
        .complexLocation()
        .build();
  }

  public String makeConfig() {
    return "test_quantize {\n" +
           " transform : linear_log_quantize\n" +
           " field1 : loc\n" +
           " scale : 10\n" +
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
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);
    assertTrue(featureVector.numFamilies() == 3);

    assertStringFamily(featureVector, "loc_quantized", 10,
                       ImmutableSet.of("a=0",
                                       "b=0.125",
                                       "c=1.125",
                                       "d=5.0",
                                       "e=17.5",
                                       "f=90",
                                       "g=350",
                                       "h=65536",
                                       "i=-1.0",
                                       "j=-23.0"));
  }
}