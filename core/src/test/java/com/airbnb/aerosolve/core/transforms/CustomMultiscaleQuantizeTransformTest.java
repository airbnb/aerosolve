package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class CustomMultiscaleQuantizeTransformTest extends BaseTransformTest {


  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .simpleStrings()
        .location()
        .sparse("loc", "zero", 0.0)
        .sparse("loc", "negative", -1.0)
        .build();
  }

  public String makeConfig() {
    return makeConfig("");
  }

  public String makeConfig(String input) {
    return "test_quantize {\n" +
        " transform : custom_multiscale_quantize\n" +
        " field1 : loc\n" + input +
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
    MultiFamilyVector featureVector = makeFeatureVector();
    transformVector(featureVector);

    assertTrue(featureVector.numFamilies() == 3);

    assertStringFamily(featureVector, "loc_quantized", 7, ImmutableSet.of(
        "lat[10.0]=30.0",
        "long[1.0]=40.0",
        "long[10.0]=40.0",
        "lat[1.0]=37.0",
        "zero=0",
        "negative[1.0]=-1.0",
        "negative[10.0]=0.0"
    ));
  }

  @Test
  public void testSelectFeatures() {
    Transform<MultiFamilyVector> transform = getTransform(makeConfig("select_features: [\"lat\"] \n"), configKey());
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);
    assertTrue(featureVector.numFamilies() == 3);

    assertStringFamily(featureVector, "loc_quantized", 2, ImmutableSet.of(
        "lat[10.0]=30.0",
        "lat[1.0]=37.0"));
  }

  @Test
  public void testExcludeFeatures() {
    Transform<MultiFamilyVector> transform = getTransform(makeConfig("exclude_features: [\"lat\"] \n"), configKey());
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);
    assertTrue(featureVector.numFamilies() == 3);

    assertStringFamily(featureVector, "loc_quantized", 5, ImmutableSet.of(
        "long[1.0]=40.0",
        "long[10.0]=40.0",
        "zero=0",
        "negative[1.0]=-1.0",
        "negative[10.0]=0.0"));
  }

  @Test
  public void testSelectAndExcludeFeatures() {
    Transform<MultiFamilyVector> transform = getTransform(makeConfig(
        "select_features: [\"lat\", \"long\"] \n" + "exclude_features: [\"lat\"] \n"), configKey());
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);
    assertTrue(featureVector.numFamilies() == 3);

    assertStringFamily(featureVector, "loc_quantized", 2, ImmutableSet.of(
        "long[1.0]=40.0",
        "long[10.0]=40.0"));
  }
}
