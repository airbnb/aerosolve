package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class CutTransformTest extends BaseTransformTest {

  public String makeConfig() {
      return "test_cut {\n" +
              " transform : cut_float\n" +
              " field1 : loc\n" +
              " upper_bound : 39.0\n" +
              " keys : [lat,long,z,aaa]\n" +
              "}";
  }

  public String makeConfigWithOutput() {
      return "test_cut {\n" +
              " transform : cut_float\n" +
              " field1 : loc\n" +
              " lower_bound : 1.0\n" +
              " upper_bound : 39.0\n" +
              " keys : [lat,long,z,aaa]\n" +
              " output : new_output \n" +
              "}";
  }

  public String makeConfigWithLowerBoundOnly() {
      return "test_cut {\n" +
              " transform : cut_float\n" +
              " field1 : loc\n" +
              " lower_bound : 1.0\n" +
              " keys : [lat,long,z,aaa]\n" +
              "}";
  }

  @Override
  public String configKey() {
    return "test_cut";
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = TransformTestingHelper.makeFoobarVector(registry);
    transform.apply(featureVector);
    assertTrue(featureVector.numFamilies() == 4);

    assertSparseFamily(featureVector, "loc", 2, ImmutableMap.of(
                           "lat", 37.7,
                           "z", -20.0),
                       ImmutableSet.of("long"));
  }

  @Test
  public void testTransformWithNewOutput() {
    Transform<MultiFamilyVector> transform = getTransform(makeConfigWithOutput(), configKey());
    MultiFamilyVector featureVector = TransformTestingHelper.makeFoobarVector(registry);
    transform.apply(featureVector);
    assertTrue(featureVector.numFamilies() == 5);

    // original feature should not change
    assertSparseFamily(featureVector, "loc", 3, ImmutableMap.of(
                           "lat", 37.7,
                           "long", 40.0,
                           "z", -20.0));

    assertSparseFamily(featureVector, "new_output", 1, ImmutableMap.of(
                           "lat", 37.7),
                       ImmutableSet.of("long", "z"));
  }

  @Test
  public void testTransformLowerBoundOnly() {
    Transform<MultiFamilyVector> transform = getTransform(makeConfigWithLowerBoundOnly(), configKey());
    MultiFamilyVector featureVector = TransformTestingHelper.makeFoobarVector(registry);
    transform.apply(featureVector);
    assertTrue(featureVector.numFamilies() == 4);

    assertSparseFamily(featureVector, "loc", 2, ImmutableMap.of(
                           "lat", 37.7,
                           "long", 40.0),
                       ImmutableSet.of("z"));
  }
}