package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.ModelRecord;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.*;
import static org.junit.Assert.assertEquals;

/**
 * Test linear function
 */

public class LinearTest {
  float func(float x) {
    return 0.2f + 1.5f * x;
  }

  Linear createLinearTestExample() {
    float [] weights = {0.2f, 1.5f};
    Linear linearFunc = new Linear(weights);
    return linearFunc;
  }

  @Test
  public void testLinearEvaluate() {
    Linear linearFunc = createLinearTestExample();
    testLinear(linearFunc);
  }

  @Test
  public void testLinearUpdate() {
    float[] weights = {0.0f, 1.0f};
    Linear linearFunc = new Linear(weights);
    Random rnd = new java.util.Random(123);

    for (int i = 0; i < 1000; i++) {
      float x = (float) (rnd.nextDouble() * 10.0 - 5.0);
      float y = func(x);
      float tmp = linearFunc.evaluate(x);
      float delta = 0.1f * (y - tmp);
      linearFunc.update(x, delta);
    }
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
  }

  void testLinear(Linear linearFunc) {
    assertEquals(0.2f, linearFunc.evaluate(0.0f), 0.01f);
    assertEquals(1.7f, linearFunc.evaluate(1.0f), 0.01f);
    assertEquals(-1.3f, linearFunc.evaluate(-1.0f), 0.01f);
  }
}
