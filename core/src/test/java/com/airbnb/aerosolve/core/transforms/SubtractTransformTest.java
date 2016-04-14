package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableMap;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class SubtractTransformTest extends BaseTransformTest {

  public String makeConfig() {
    return "test_subtract {\n" +
        " transform : subtract\n" +
        " field1 : loc\n" +
        " field2 : F\n" +
        " keys : [\"lat\"] \n" +
        " key2 : foo\n" +
        " output : bar\n" +
        "}";
  }

  @Override
  public String configKey() {
    return "test_subtract";
  }

  @Test
  public void testTransformWithKeys() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = TransformTestingHelper.makeFoobarVector(registry);
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 4);

    assertSparseFamily(featureVector, "bar", 2,
                       ImmutableMap.of("lat-foo", 36.2,
                                       "bar_fv", 1.0));
  }
}