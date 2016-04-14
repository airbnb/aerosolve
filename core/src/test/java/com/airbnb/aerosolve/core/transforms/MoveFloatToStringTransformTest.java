package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class MoveFloatToStringTransformTest extends BaseTransformTest {

  public String makeConfig(boolean moveAllKeys) {
    StringBuilder sb = new StringBuilder();
    sb.append("test_move_float_to_string {\n");
    sb.append(" transform : move_float_to_string\n");
    sb.append(" field1 : loc\n");
    sb.append(" bucket : 1\n");

    if (!moveAllKeys) {
      sb.append(" keys : [lat]\n");
    }

    sb.append(" output : loc_quantized\n");
    sb.append("}");
    return sb.toString();
  }

  @Override
  public String configKey() {
    return "test_move_float_to_string";
  }

  @Override
  public String makeConfig() {
    return makeConfig(true);
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = TransformTestingHelper.makeFoobarVector(registry);
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 5);

    assertStringFamily(featureVector, "loc_quantized", 1,
                       ImmutableSet.of("lat=37.0"));
  }

  @Test
  public void testTransformMoveAllKeys() {
    Transform<MultiFamilyVector> transform = getTransform(makeConfig(false), configKey());
    MultiFamilyVector featureVector = TransformTestingHelper.makeFoobarVector(registry);
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 5);

    assertStringFamily(featureVector, "loc_quantized", 3,
                       ImmutableSet.of("lat=37.0", "long=40.0", "z=-20.0"));
  }
}