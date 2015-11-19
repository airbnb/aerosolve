package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.ModelRecord;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.*;
import static org.junit.Assert.assertEquals;

public class PolynomialTest {
  private static final Logger log = LoggerFactory.getLogger(SplineTest.class);
  Polynomial createPolynomialTestExample() {
    // create a test case: f(x) = 1 + 2 * x + 4 * x * x
    float[] weights = new float[3];
    weights[0] = 1.0f;
    weights[1] = 2.0f;
    weights[2] = 4.0f;
    return new Polynomial(weights);
  }

  float evalPolyExample(float x) {
    return 1.0f + 2.0f * x + 4.0f * x * x;
  }

  @Test
  public void testPolynomialEvaluate() {
    Polynomial poly = createPolynomialTestExample();
    testPolynomial(poly);
  }

  @Test
  public void testPolynomialModelRecordConstructor() {
    ModelRecord record = new ModelRecord();
    record.setFeatureFamily("TEST");
    record.setFeatureName("a");
    List<Double> weightVec = new ArrayList<Double>();
    weightVec.add(1.0);
    weightVec.add(2.0);
    weightVec.add(4.0);
    record.setWeightVector(weightVec);
    Polynomial poly = new Polynomial(record);
    testPolynomial(poly);
  }

  @Test
  public void testPolynomialToModelRecord() {
    Polynomial poly = createPolynomialTestExample();
    ModelRecord record = poly.toModelRecord("family", "name");
    assertEquals(record.getFeatureFamily(), "family");
    assertEquals(record.getFeatureName(), "name");
    List<Double> weightVector = record.getWeightVector();
    assertEquals(1.0f, weightVector.get(0).floatValue(), 0.01f);
    assertEquals(2.0f, weightVector.get(1).floatValue(), 0.01f);
    assertEquals(4.0f, weightVector.get(2).floatValue(), 0.01f);
  }

  void testPolynomial(Polynomial poly) {
    assertEquals(evalPolyExample(-10.0f), poly.evaluate(-10.0f), 0.01f);
    assertEquals(evalPolyExample(0.0f), poly.evaluate(0.0f), 0.01f);
    assertEquals(evalPolyExample(1.0f), poly.evaluate(1.0f), 0.01f);
  }

  void logPolyWeights(Polynomial poly) {
    int n = poly.getWeights().length;
    for (int i = 0; i < n; i++) {
      log.info("weight" + poly.getWeights()[i]);
    }
  }

}
