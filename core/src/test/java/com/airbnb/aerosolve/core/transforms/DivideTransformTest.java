package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableMap;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class DivideTransformTest extends BaseTransformTest {

  public String makeConfig() {
    return "test_divide {\n" +
           " transform : divide\n" +
           " field1 : loc\n" +
           " field2 : F\n" +
           " keys : [ lat, long ] \n" +
           " key2 : foo\n" +
           " constant : 0.1\n" +
           " output : bar\n" +
           "}";
  }

  @Override
  public String configKey() {
    return "test_divide";
  }

  @Test
  public void testTransformWithKeys() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = TransformTestingHelper.makeFoobarVector(registry);
    transform.apply(featureVector);
    assertTrue(featureVector.numFamilies() == 4);

    assertSparseFamily(featureVector, "bar", 3, ImmutableMap.of(
        "bar_fv", 1.0,
        "lat-d-foo", 37.7 / 1.6,
        "long-d-foo", 40.0 / 1.6
    ));
  }
}