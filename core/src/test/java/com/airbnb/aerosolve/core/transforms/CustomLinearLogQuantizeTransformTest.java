package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

@Slf4j
public class CustomLinearLogQuantizeTransformTest extends BaseTransformTest{
  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .simpleStrings()
        .complexLocation()
        .build();
  }

  public String makeConfig() {
    return "test_quantize {\n" +
        " transform : custom_linear_log_quantize\n" +
        " field1 : loc\n" +
        " scale : 10\n" +
        " output : loc_quantized\n" +
        " limit_bucket: [{\"1.0\" : \"0.125\"},\n" +
        "    {\"10.0\" : \"0.5\"},\n" +
        "    {\"25.0\" : \"2.0\"}\n" +
        "    {\"50.0\" : \"5.0\"},\n" +
        "    {\"100.0\" : \"10.0\"},\n" +
        "    {\"400.0\" : \"25.0\"},\n" +
        "    {\"2000.0\" : \"100.0\"},\n" +
        "    {\"10000.0\" : \"250.0\"}\n" +
        "  ]" +
        "}";
  }

  @Override
  public String configKey() {
    return "test_quantize";
  }

  @Test
  public void testTransform() {
    MultiFamilyVector featureVector = transformVector(makeFeatureVector());

    assertTrue(featureVector.numFamilies() == 3);

    assertStringFamily(featureVector, "loc_quantized", 10,
                       ImmutableSet.of("a=0.0",
                                       "b=0.125",
                                       "c=1.0",
                                       "d=5.0",
                                       "e=16.0",
                                       "f=90.0",
                                       "g=350.0",
                                       "h=10000.0",
                                       "i=-1.0",
                                       "j=-22.0"));
  }
}