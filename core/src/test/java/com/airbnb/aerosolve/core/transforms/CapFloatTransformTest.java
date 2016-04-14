package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableMap;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class CapFloatTransformTest extends BaseTransformTest {
  public String makeConfig() {
    return "test_cap {\n" +
           " transform : cap_float\n" +
           " field1 : loc\n" +
           " lower_bound : 1.0\n" +
           " upper_bound : 39.0\n" +
           " keys : [lat,long,z,aaa]\n" +
           "}";
  }

  public String makeConfigWithOutput() {
    return "test_cap {\n" +
        " transform : cap_float\n" +
        " field1 : loc\n" +
        " lower_bound : 1.0\n" +
        " upper_bound : 39.0\n" +
        " keys : [lat,long,z,aaa]\n" +
        " output : new_output \n" +
        "}";
  }

  @Override
  public String configKey() {
    return "test_cap";
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = TransformTestingHelper.makeFoobarVector(registry);
    transform.apply(featureVector);
    assertTrue(featureVector.numFamilies() == 4);

    assertSparseFamily(featureVector, "loc", 3,
                       ImmutableMap.of("lat", 37.7,
                                       "long", 39.0,
                                       "z", 1.0));
  }

  @Test
  public void testTransformWithNewOutput() {
    Transform<MultiFamilyVector> transform = getTransform(makeConfigWithOutput(), "test_cap");
    MultiFamilyVector featureVector = TransformTestingHelper.makeFoobarVector(registry);
    transform.apply(featureVector);
    assertTrue(featureVector.numFamilies() == 5);

    // original feature should not change
    assertSparseFamily(featureVector, "loc", 3,
                       ImmutableMap.of("lat", 37.7,
                                       "long", 40.0,
                                       "z", -20.0));

    // capped features are in a new feature family
    assertSparseFamily(featureVector, "new_output", 3,
                       ImmutableMap.of("lat", 37.7,
                                       "long", 39.0,
                                       "z", 1.0));
  }
}