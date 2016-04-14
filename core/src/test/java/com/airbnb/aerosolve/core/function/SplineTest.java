package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.ModelRecord;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import static org.junit.Assert.assertEquals;

/**
 * @author Hector Yee
 */
@Slf4j
public class SplineTest {
  @Test
  public void testSplineEvaluate() {
    double[] weights = {5.0d, 10.0d, -20.0d};
    Spline spline = new Spline(1.0d, 3.0d, weights);
    testSpline(spline, 0.1d);
  }

  public static Spline getSpline() {
    double[] weights = {5.0d, 10.0d, -20.0d};
    return new Spline(1.0d, 3.0d, weights);
  }

  @Test
  public void testSplineResampleConstructor() {
    Spline spline = getSpline();

    // Same size
    spline.resample(3);
    testSpline(spline, 0.1d);

    // Smaller
    Spline spline3 = getSpline();
    spline3.resample(2);
    assertEquals(5.0d, spline3.evaluate(-1.0d), 0.1d);
    assertEquals(5.0d, spline3.evaluate(1.0d), 0.1d);
    assertEquals((5.0d - 20.0d) * 0.5d, spline3.evaluate(2.0d), 0.1d);
    assertEquals(-20.0d, spline3.evaluate(3.0d), 0.1d);
    assertEquals(-20.0d, spline3.evaluate(4.0d), 0.1d);
    
    // Larger
    Spline spline4 = getSpline();
    spline4.resample(100);
    testSpline(spline4, 0.2d);
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
    testSpline(spline, 0.1d);
  }

  @Test
  public void testSplineToModelRecord() {
    double[] weights = {5.0d, 10.0d, -20.0d};
    Spline spline = new Spline(1.0d, 3.0d, weights);
    ModelRecord record = spline.toModelRecord("family", "name");
    assertEquals(record.getFeatureFamily(), "family");
    assertEquals(record.getFeatureName(), "name");
    List<Double> weightVector = record.getWeightVector();
    assertEquals(5.0d, weightVector.get(0), 0.01d);
    assertEquals(10.0d, weightVector.get(1), 0.01d);
    assertEquals(-20.0d, weightVector.get(2), 0.01d);
    assertEquals(1.0d, record.getMinVal(), 0.01d);
    assertEquals(3.0d, record.getMaxVal(), 0.01d);
  }

  @Test
  public void testSplineResample() {
    double[] weights = {5.0d, 10.0d, -20.0d};
    // Same size
    Spline spline1 = new Spline(1.0d, 3.0d, weights);
    spline1.resample(3);
    testSpline(spline1, 0.1d);

    // Smaller
    Spline spline2 = new Spline(1.0d, 3.0d, weights);
    spline2.resample(2);
    assertEquals(5.0d, spline2.evaluate(-1.0d), 0.1d);
    assertEquals(5.0d, spline2.evaluate(1.0d), 0.1d);
    assertEquals((5.0d - 20.0d) * 0.5d, spline2.evaluate(2.0d), 0.1d);
    assertEquals(-20.0d, spline2.evaluate(3.0d), 0.1d);
    assertEquals(-20.0d, spline2.evaluate(4.0d), 0.1d);

    // Larger
    Spline spline3 = new Spline(1.0d, 3.0d, weights);
    spline3.resample(100);
    testSpline(spline3, 0.2d);
    spline3.resample(200);
    testSpline(spline3, 0.2d);
  }
  
  void testSpline(Spline spline, double tol) {
    double a = spline.evaluate(1.5d);
    log.info("spline 1.5 is " + a);
    assertEquals(5.0d, spline.evaluate(-1.0d), tol);
    assertEquals(5.0d, spline.evaluate(1.0d), tol);
    assertEquals(7.5f, spline.evaluate(1.5d), tol);
    assertEquals(10.0d, spline.evaluate(1.99d), tol);
    assertEquals(10.0d, spline.evaluate(2.0d), tol);
    assertEquals(0.0d, spline.evaluate(2.3333d), tol);
    assertEquals(-10.0d, spline.evaluate(2.667d), tol);
    assertEquals(-20.0d, spline.evaluate(2.99999d), tol);
    assertEquals(-20.0d, spline.evaluate(3.0d), tol);
    assertEquals(-20.0d, spline.evaluate(4.0d), tol);
  }

  double func(double x) {
    return 0.1d * (x + 0.5d) * (x - 4.0d) * (x - 1.0d);
  }

  @Test
  public void testSplineUpdate() {
    double[] weights = new double[8];
    Spline spline = new Spline(-1.0d, 5.0d, weights);
    Random rnd = new java.util.Random(123);
    for (int i = 0; i < 1000; i++) {
      double x = (double) (rnd.nextDouble() * 6.0 - 1.0);
      double y = func(x);
      double tmp = spline.evaluate(x);
      double delta =0.1d * (y - tmp);
      spline.update(x, delta);
    }
    // Check we get roots where we expect them to be.
    assertEquals(0.0d, spline.evaluate(-0.5d), 0.1d);
    assertEquals(0.0d, spline.evaluate(1.0d), 0.1d);
    assertEquals(0.0d, spline.evaluate(4.0d), 0.1d);
    for (int i = 0; i < 20; i++) {
      double x = (double) (6.0 * i / 20.0 - 1.0d);
      double expected = func(x);
      double eval = spline.evaluate(x);
      log.info("x = " + x + " expected = " + expected + " got = " + eval);
      assertEquals(expected, spline.evaluate(x), 0.1d);
    }
  }

  @Test
  public void testSplineL1Norm() {
    double[] weights1 = {5.0d, 10.0d, -20.0d};
    Spline spline1 = new Spline(1.0d, 3.0d, weights1);
    assertEquals(35.0d, spline1.L1Norm(), 0.01d);

    double[] weights2 = {0.0d, 0.0d};
    Spline spline2 = new Spline(1.0d, 3.0d, weights2);
    assertEquals(0.0d, spline2.L1Norm(), 0.01d);
  }

  @Test
  public void testSplineLInfinityNorm() {
    double[] weights1 = {5.0d, 10.0d, -20.0d};
    Spline spline1 = new Spline(1.0d, 3.0d, weights1);
    assertEquals(20.0d, spline1.LInfinityNorm(), 0.01d);

    double[] weights2 = {0.0d, 0.0d};
    Spline spline2 = new Spline(1.0d, 3.0d, weights2);
    assertEquals(0.0d, spline2.LInfinityNorm(), 0.01d);
  }

  @Test
  public void testSplineLInfinityCap() {
    double[] weights = {5.0d, 10.0d, -20.0d};
    Spline spline1 = new Spline(1.0d, 3.0d, weights);
    // Larger (no scale)
    spline1.LInfinityCap(30.0d);
    assertEquals(5.0d, spline1.getWeights()[0], 0.01d);
    assertEquals(10.0d, spline1.getWeights()[1], 0.01d);
    assertEquals(-20.0d, spline1.getWeights()[2], 0.01d);
    // Negative
    spline1.LInfinityCap(-10.0d);
    assertEquals(5.0d, spline1.getWeights()[0], 0.01d);
    assertEquals(10.0d, spline1.getWeights()[1], 0.01d);
    assertEquals(-20.0d, spline1.getWeights()[2], 0.01d);
    // Smaller (with scale)
    Spline spline2 = new Spline(1.0d, 3.0d, weights);
    spline2.LInfinityCap(10.0d);
    double scale = 10.0d / 20.0d;
    assertEquals(5.0d * scale, spline2.getWeights()[0], 0.01d);
    assertEquals(10.0d * scale, spline2.getWeights()[1], 0.01d);
    assertEquals(-20.0d * scale, spline2.getWeights()[2], 0.01d);
  }
}
