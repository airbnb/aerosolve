package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class FloatVectorTest {
  private static final Logger log = LoggerFactory.getLogger(FloatVectorTest.class);

  @Test
  public void testDot() {
    FloatVector v1 = new FloatVector(new float[]{1.0f, 2.0f});
    FloatVector v2 = new FloatVector(new float[]{3.0f, 4.0f});
    assertEquals(1.0f * 3.0f + 2.0f * 4.0f, v1.dot(v2), 0.1f);
  }

  @Test
  public void testMultiplyAdd() {
    FloatVector v1 = new FloatVector(new float[]{1.0f, 2.0f});
    FloatVector v2 = new FloatVector(new float[]{3.0f, 4.0f});
    v1.multiplyAdd(2.0f, v2);
    assertEquals(1.0 + 2.0 * 3.0, v1.values[0], 0.1);
    assertEquals(2.0 + 2.0 * 4.0, v1.values[1], 0.1);
  }

  @Test
  public void testRectify() {
    FloatVector v1 = new FloatVector(new float[]{1.0f, -2.0f});
    v1.rectify();
    assertEquals(1.0f, v1.values[0], 0.1f);
    assertEquals(0.0f, v1.values[1], 0.1f);
  }

  @Test
  public void testHadamard() {
    FloatVector v1 = new FloatVector(new float[]{1.0f, 2.0f});
    FloatVector v2 = new FloatVector(new float[]{3.0f, 4.0f});
    FloatVector v3 = FloatVector.Hadamard(v1, v2);
    assertEquals(3.0, v3.values[0], 0.1);
    assertEquals(8.0, v3.values[1], 0.1);
  }

  @Test
  public void textMinMaxResult() {
    FloatVector v = new FloatVector(new float[]{1.0f, -2.0f, 3.0f, 4.0f});
    FloatVector.MinMaxResult result = v.getMinMaxResult();
    assertEquals(0, result.minIndex);
    assertEquals(1.0, result.minValue, 0.1f);

    assertEquals(3, result.maxIndex);
    assertEquals(4.0, result.maxValue, 0.1f);
  }

}