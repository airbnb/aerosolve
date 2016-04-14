package com.airbnb.aerosolve.core.functions;

import com.airbnb.aerosolve.core.ModelRecord;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Test linear function
 */

public class LinearTest {
  double func(double x) {
    return 0.2d + 1.5d * (x + 6.0d) / 11.0d;
  }

  Linear createLinearTestExample() {
    double [] weights = {0.2d, 1.5d};
    return new Linear(-6.0d, 5.0d, weights);
  }

  @Test
  public void testLinearEvaluate() {
    Linear linearFunc = createLinearTestExample();
    testLinear(linearFunc);
  }

  @Test
  public void testLinearModelRecordConstructor() {
    ModelRecord record = new ModelRecord();
    record.setFeatureFamily("TEST");
    record.setFeatureName("a");
    List<Double> weightVec = new ArrayList<Double>();
    weightVec.add(0.2);
    weightVec.add(1.5);
    record.setWeightVector(weightVec);
    record.setMinVal(-6.0d);
    record.setMaxVal(5.0d);
    Linear linearFunc = new Linear(record);
    testLinear(linearFunc);
  }

  @Test
  public void testLinearToModelRecord() {
    Linear linearFunc = createLinearTestExample();
    ModelRecord record = linearFunc.toModelRecord("family", "name");
    assertEquals(record.getFeatureFamily(), "family");
    assertEquals(record.getFeatureName(), "name");
    List<Double> weightVector = record.getWeightVector();
    assertEquals(0.2d, weightVector.get(0), 0.01d);
    assertEquals(1.5d, weightVector.get(1), 0.01d);
    assertEquals(-6.0d, record.getMinVal(), 0.01d);
    assertEquals(5.0d, record.getMaxVal(), 0.01d);
  }

  @Test
  public void testLinearUpdate() {
    Linear linearFunc = new Linear(-6.0d, 5.0d, new double[2]);
    Random rnd = new java.util.Random(123);
    for (int i = 0; i < 1000; i++) {
      double x = (rnd.nextDouble() * 10.0 - 5.0);
      double y = func(x);
      double tmp = linearFunc.evaluate(x);
      double delta = 0.5d * (y - tmp);
      linearFunc.update(delta, x);
    }
    testLinear(linearFunc);
  }

  void testLinear(Linear linearFunc) {
    assertEquals(0.2d + 1.5d * 6.0d / 11.0d, linearFunc.evaluate(0.0d), 0.01d);
    assertEquals(0.2d + 1.5d * 7.0d / 11.0d, linearFunc.evaluate(1.0d), 0.01d);
    assertEquals(0.2d + 1.5d * 5.0d / 11.0d, linearFunc.evaluate(-1.0d), 0.01d);
  }
}
