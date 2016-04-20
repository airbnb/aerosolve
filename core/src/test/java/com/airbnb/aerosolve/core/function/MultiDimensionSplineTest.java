package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.models.NDTreeModel;
import com.airbnb.aerosolve.core.models.NDTreeModelTest;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

@Slf4j
public class MultiDimensionSplineTest {
  public MultiDimensionSpline getMultiDimensionSpline() {
    NDTreeModel tree = NDTreeModelTest.getNDTreeModel();
    return new MultiDimensionSpline(tree);
  }

  @Test
  public void evaluate() throws Exception {
    MultiDimensionSpline spline = getMultiDimensionSpline();
    spline.update(Arrays.asList(0.5, 2.0), 0.5);
    assertEquals(0.125, spline.evaluate(Arrays.asList(0.5, 2.0)), 0.0);

    spline.update(Arrays.asList(0.6, 2.0), 0.8);
    assertEquals(0.325, spline.evaluate(Arrays.asList(0.5, 2.0)), 0.0);

    spline.update(Arrays.asList(2.0, 1.0), 0.8);
    assertEquals(0.28868586584843375, spline.evaluate(Arrays.asList(3.0, 1.0)), 0.0);

    spline.update(Arrays.asList(3.0, 3.0), 0.8);
    assertEquals(0.40389338302461303, spline.evaluate(Arrays.asList(3.0, 3.0)), 0.0);
  }
}