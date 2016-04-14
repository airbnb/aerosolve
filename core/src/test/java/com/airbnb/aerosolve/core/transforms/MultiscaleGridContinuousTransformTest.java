package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.google.common.collect.ImmutableMap;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class MultiscaleGridContinuousTransformTest extends BaseTransformTest {

  public String makeConfig() {
    return "test_grid {\n" +
           " transform : multiscale_grid_continuous\n" +
           " field1 : loc\n" +
           " value1 : lat\n" +
           " value2 : long\n" +
           " buckets : [1, 5]\n" +
           " output : loc_continuous\n" +
           "}";
  }

  @Override
  public String configKey() {
    return "test_grid";
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = TransformTestingHelper.makeSimpleVector(registry);
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 3);

    assertSparseFamily(featureVector, "loc_continuous", 4, ImmutableMap.of(
        "[1.0]=(37.0,40.0)@1", 0.7,
        "[5.0]=(35.0,40.0)@1", 2.7,
        "[1.0]=(37.0,40.0)@2", 0.0,
        "[5.0]=(35.0,40.0)@2", 0.0
    ));
  }
}