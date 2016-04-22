package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.models.NDTreeModel;
import com.airbnb.aerosolve.core.models.NDTreeModelTest;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

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
    eval(spline);
  }

  private static void eval(Function spline) {
    spline.update((float)0.5, (float)0.5, (float)2.0);
    assertEquals(0.125, spline.evaluate((float) 0.5, (float)2.0), 0.0001);

    spline.update((float)0.8, (float)0.6, (float)2.0);
    assertEquals(0.325, spline.evaluate((float)0.5, (float)2.0), 0.0001);

    spline.update((float) 0.8, (float)2.0, (float)1.0);
    assertEquals(0.28868586584843375, spline.evaluate((float)3.0, (float)1.0), 0.0001);

    spline.update((float)0.8, (float)3.0, (float)3.0);
    assertEquals(0.40389338302461303, spline.evaluate((float)3.0, (float)3.0), 0.0001);

  }

  @Test
  public void aggregate() throws Exception {
    List<Function> splineList = new ArrayList<>();
    for (int i = 0; i < 10; ++i) {
      MultiDimensionSpline spline = getMultiDimensionSpline();
      splineList.add(spline);
    }
    Function r = splineList.get(0).aggregate(splineList, 0.1f, 0);
    eval(r);
  }
}