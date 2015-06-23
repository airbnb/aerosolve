package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class SplineTest {
  private static final Logger log = LoggerFactory.getLogger(SplineTest.class);

  @Test
  public void testSplineEvaluate() {
    float[] weights = {5.0f, 10.0f, -20.0f};
    Spline spline = new Spline(1.0f, 3.0f, weights);
    testSpline(spline, 0.1f);
  }
  
  @Test
  public void testSplineResample() {
    float[] weights = {5.0f, 10.0f, -20.0f};
    Spline spline = new Spline(1.0f, 3.0f, weights);

    // Same size
    Spline spline2 = new Spline(spline, 3);
    testSpline(spline2, 0.1f);
    
    // Smaller
    Spline spline3 = new Spline(spline, 2);
    assertEquals(5.0f, spline3.evaluate(-1.0f), 0.1f);
    assertEquals(5.0f, spline3.evaluate(1.0f), 0.1f);
    assertEquals((5.0f - 20.0f) * 0.5f, spline3.evaluate(2.0f), 0.1f);
    assertEquals(-20.0f, spline3.evaluate(3.0f), 0.1f);
    assertEquals(-20.0f, spline3.evaluate(4.0f), 0.1f);
    
    // Larger
    Spline spline4 = new Spline(spline, 100);
    testSpline(spline4, 0.2f);
  }
  
  void testSpline(Spline spline, float tol) {
    float a = spline.evaluate(1.5f);
    log.info("spline 1.5 is " + a);
    assertEquals(5.0f, spline.evaluate(-1.0f), tol);
    assertEquals(5.0f, spline.evaluate(1.0f), tol);
    assertEquals(7.5f, spline.evaluate(1.5f), tol);
    assertEquals(10.0f, spline.evaluate(1.99f), tol);
    assertEquals(10.0f, spline.evaluate(2.0f), tol);
    assertEquals(0.0f, spline.evaluate(2.3333f), tol);
    assertEquals(-10.0f, spline.evaluate(2.667f), tol);
    assertEquals(-20.0f, spline.evaluate(2.99999f), tol);
    assertEquals(-20.0f, spline.evaluate(3.0f), tol);
    assertEquals(-20.0f, spline.evaluate(4.0f), tol);
  }

  float func(float x) {
    return 0.1f * (x + 0.5f) * (x - 4.0f) * (x - 1.0f);
  }

  @Test
  public void testSplineUpdate() {
    float[] weights = new float[8];
    Spline spline = new Spline(-1.0f, 5.0f, weights);
    Random rnd = new java.util.Random(123);
    for (int i = 0; i < 1000; i++) {
      float x = (float) (rnd.nextDouble() * 6.0 - 1.0);
      float y = func(x);
      float tmp = spline.evaluate(x);
      float delta =0.1f * (y - tmp);
      spline.update(x, delta);
    }
    // Check we get roots where we expect them to be.
    assertEquals(0.0f, spline.evaluate(-0.5f), 0.1f);
    assertEquals(0.0f, spline.evaluate(1.0f), 0.1f);
    assertEquals(0.0f, spline.evaluate(4.0f), 0.1f);
    for (int i = 0; i < 20; i++) {
      float x = (float) (6.0 * i / 20.0 - 1.0f);
      float expected = func(x);
      float eval = spline.evaluate(x);
      log.info("x = " + x + " expected = " + expected + " got = " + eval);
      assertEquals(expected, spline.evaluate(x), 0.1f);
    }
  }
}
