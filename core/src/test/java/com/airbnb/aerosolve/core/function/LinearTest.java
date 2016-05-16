package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.ModelRecord;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Test linear function
 */

public class LinearTest {
  private float func(float x) {
    return 0.2f + 1.5f * (x + 6.0f) / 11.0f;
  }

  private Linear createLinearTestExample() {
    float [] weights = {0.2f, 1.5f};
    return new Linear(-6.0f, 5.0f, weights);
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
    List<Double> weightVec = new ArrayList<>();
    weightVec.add(0.2);
    weightVec.add(1.5);
    record.setWeightVector(weightVec);
    record.setMinVal(-6.0f);
    record.setMaxVal(5.0f);
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
    assertEquals(0.2f, weightVector.get(0).floatValue(), 0.01f);
    assertEquals(1.5f, weightVector.get(1).floatValue(), 0.01f);
    assertEquals(-6.0f, record.getMinVal(), 0.01f);
    assertEquals(5.0f, record.getMaxVal(), 0.01f);
  }

  @Test
  public void testLinearUpdate() {
    Linear linearFunc = new Linear(-6.0f, 5.0f, new float[2]);
    Random rnd = new java.util.Random(123);
    for (int i = 0; i < 1000; i++) {
      float x = (float) (rnd.nextDouble() * 10.0 - 5.0);
      float y = func(x);
      float tmp = linearFunc.evaluate(x);
      float delta = 0.5f * (y - tmp);
      linearFunc.update(delta, x);
    }
    testLinear(linearFunc);
  }

  @Test
  public void testLinearClone() throws CloneNotSupportedException {
    Linear linearFunc = new Linear(-6.0f, 5.0f, new float[2]);
    Linear linearCopy = linearFunc.clone();
    linearCopy.weights = new float[]{1.0f, 2.0f, 3.0f};
    assertArrayEquals(linearFunc.weights, new float[2], 0);
    linearCopy.minVal = 0f;
    assertEquals(linearFunc.minVal, -6f, 0);
  }

  private void testLinear(Linear linearFunc) {
    assertEquals(0.2f + 1.5f * 6.0f / 11.0f, linearFunc.evaluate(0.0f), 0.01f);
    assertEquals(0.2f + 1.5f * 7.0f / 11.0f, linearFunc.evaluate(1.0f), 0.01f);
    assertEquals(0.2f + 1.5f * 5.0f / 11.0f, linearFunc.evaluate(-1.0f), 0.01f);
  }
}
