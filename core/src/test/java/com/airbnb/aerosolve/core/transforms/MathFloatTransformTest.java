package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import org.junit.Test;

import static org.junit.Assert.assertTrue;
public class MathFloatTransformTest extends BaseTransformTest {

  public String makeConfig() {
    return makeConfig("log10");
  }

  public String makeConfig(String functionName) {
    return "test_math {\n" +
        " transform : math_float\n" +
        " field1 : loc\n" +
        " keys : [lat,long,z]\n" +
        " output : bar\n" +
        " function : " + functionName +
        "}";
  }

  @Override
  public String configKey() {
    return "test_math";
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = TransformTestingHelper.makeFoobarVector(registry);
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 4);

    // the original features are unchanged
    assertSparseFamily(featureVector, "loc", 3,
                       ImmutableMap.of("lat", 37.7,
                                       "long", 40.0,
                                       "z", -20.0));

    assertSparseFamily(featureVector, "bar", 3,
                       ImmutableMap.of("lat", Math.log10(37.7),
                                       "long", Math.log10(40.0),
                                       // existing feature in 'bar' should not change
                                       "bar_fv", 1.0),
                       // for negative value, it would be a missing feature
                       ImmutableSet.of("z"));
  }

  @Test(expected = IllegalArgumentException.class)
  public void testUndefinedFunction() {
    Transform<MultiFamilyVector> transform = getTransform(makeConfig("tan"), configKey());
    MultiFamilyVector featureVector = TransformTestingHelper.makeFoobarVector(registry);
    transform.apply(featureVector);

    // TODO (Brad): I made this throw an exception.  Does it really make sense to just continue
    // as usual doing nothing if the function is unknown?
    assertTrue(featureVector.numFamilies() == 4);

    // the original features are unchanged
    assertSparseFamily(featureVector, "loc", 3,
                       ImmutableMap.of("lat", 37.7,
                                       "long", 40.0,
                                       "z", -20.0));
    // new features should not exist
    assertSparseFamily(featureVector, "bar", 1,
                       ImmutableMap.of("bar_fv", 1.0));
  }
}
