package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.ModelRecord;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertEquals;

/**
 * @author Hector Yee
 */
@Slf4j
public class SplineTest {
  @Test
  public void testSplineEvaluate() {
    float[] weights = {5.0f, 10.0f, -20.0f};
    Spline spline = new Spline(1.0f, 3.0f, weights);
    testSpline(spline, 0.1f);
  }

  public static Spline getSpline() {
    float[] weights = {5.0f, 10.0f, -20.0f};
    return new Spline(1.0f, 3.0f, weights);

  }
  @Test
  public void testSplineResampleConstructor() {
    Spline spline = getSpline();

    // Same size
    spline.resample(3);
    testSpline(spline, 0.1f);
    
    // Smaller
    Spline spline3 = getSpline();
    spline3.resample(2);
    assertEquals(5.0f, spline3.evaluate(-1.0f), 0.1f);
    assertEquals(5.0f, spline3.evaluate(1.0f), 0.1f);
    assertEquals((5.0f - 20.0f) * 0.5f, spline3.evaluate(2.0f), 0.1f);
    assertEquals(-20.0f, spline3.evaluate(3.0f), 0.1f);
    assertEquals(-20.0f, spline3.evaluate(4.0f), 0.1f);
    
    // Larger
    Spline spline4 = getSpline();
    spline4.resample(100);
    testSpline(spline4, 0.2f);
  }

  @Test
  public void testSplineModelRecordConstructor() {
    ModelRecord record = new ModelRecord();
    record.setFeatureFamily("TEST");
    record.setFeatureName("a");
    record.setMinVal(1.0);
    record.setMaxVal(3.0);
    List<Double> weightVec = new ArrayList<Double>();
    weightVec.add(5.0);
    weightVec.add(10.0);
    weightVec.add(-20.0);
    record.setWeightVector(weightVec);
    Spline spline = new Spline(record);
    testSpline(spline, 0.1f);
  }

  @Test
  public void testSplineToModelRecord() {
    float[] weights = {5.0f, 10.0f, -20.0f};
    Spline spline = new Spline(1.0f, 3.0f, weights);
    ModelRecord record = spline.toModelRecord("family", "name");
    assertEquals(record.getFeatureFamily(), "family");
    assertEquals(record.getFeatureName(), "name");
    List<Double> weightVector = record.getWeightVector();
    assertEquals(5.0f, weightVector.get(0).floatValue(), 0.01f);
    assertEquals(10.0f, weightVector.get(1).floatValue(), 0.01f);
    assertEquals(-20.0f, weightVector.get(2).floatValue(), 0.01f);
    assertEquals(1.0f, record.getMinVal(), 0.01f);
    assertEquals(3.0f, record.getMaxVal(), 0.01f);
  }

  @Test
  public void testSplineResample() {
    float[] weights = {5.0f, 10.0f, -20.0f};
    // Same size
    Spline spline1 = new Spline(1.0f, 3.0f, weights);
    spline1.resample(3);
    testSpline(spline1, 0.1f);

    // Smaller
    Spline spline2 = new Spline(1.0f, 3.0f, weights);
    spline2.resample(2);
    assertEquals(5.0f, spline2.evaluate(-1.0f), 0.1f);
    assertEquals(5.0f, spline2.evaluate(1.0f), 0.1f);
    assertEquals((5.0f - 20.0f) * 0.5f, spline2.evaluate(2.0f), 0.1f);
    assertEquals(-20.0f, spline2.evaluate(3.0f), 0.1f);
    assertEquals(-20.0f, spline2.evaluate(4.0f), 0.1f);

    // Larger
    Spline spline3 = new Spline(1.0f, 3.0f, weights);
    spline3.resample(100);
    testSpline(spline3, 0.2f);
    spline3.resample(200);
    testSpline(spline3, 0.2f);
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
      spline.update(delta, x);
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

  @Test
  public void testSplineL1Norm() {
    float[] weights1 = {5.0f, 10.0f, -20.0f};
    Spline spline1 = new Spline(1.0f, 3.0f, weights1);
    assertEquals(35.0f, spline1.L1Norm(), 0.01f);

    float[] weights2 = {0.0f, 0.0f};
    Spline spline2 = new Spline(1.0f, 3.0f, weights2);
    assertEquals(0.0f, spline2.L1Norm(), 0.01f);
  }

  @Test
  public void testSplineLInfinityNorm() {
    float[] weights1 = {5.0f, 10.0f, -20.0f};
    Spline spline1 = new Spline(1.0f, 3.0f, weights1);
    assertEquals(20.0f, spline1.LInfinityNorm(), 0.01f);

    float[] weights2 = {0.0f, 0.0f};
    Spline spline2 = new Spline(1.0f, 3.0f, weights2);
    assertEquals(0.0f, spline2.LInfinityNorm(), 0.01f);
  }

  @Test
  public void testSplineLInfinityCap() {
    float[] weights = {5.0f, 10.0f, -20.0f};
    Spline spline1 = new Spline(1.0f, 3.0f, weights);
    // Larger (no scale)
    spline1.LInfinityCap(30.0f);
    assertEquals(5.0f, spline1.getWeights()[0], 0.01f);
    assertEquals(10.0f, spline1.getWeights()[1], 0.01f);
    assertEquals(-20.0f, spline1.getWeights()[2], 0.01f);
    // Negative
    spline1.LInfinityCap(-10.0f);
    assertEquals(5.0f, spline1.getWeights()[0], 0.01f);
    assertEquals(10.0f, spline1.getWeights()[1], 0.01f);
    assertEquals(-20.0f, spline1.getWeights()[2], 0.01f);
    // Smaller (with scale)
    Spline spline2 = new Spline(1.0f, 3.0f, weights);
    spline2.LInfinityCap(10.0f);
    float scale = 10.0f / 20.0f;
    assertEquals(5.0f * scale, spline2.getWeights()[0], 0.01f);
    assertEquals(10.0f * scale, spline2.getWeights()[1], 0.01f);
    assertEquals(-20.0f * scale, spline2.getWeights()[2], 0.01f);
  }
}
