package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.ModelRecord;
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
      set(spline);
      splineList.add(spline);
    }
    Function r = splineList.get(0).aggregate(splineList, 0.1f, 0);
    assertEquals(0.40389338302461303, r.evaluate(3.0f, 3.0f), 0.0001);
  }

  private static void set(Function spline) {
    spline.update(0.5f, 0.5f, 2.0f);
    spline.update(0.8f, 0.6f, 2.0f);
    spline.update(0.8f, 2.0f, 1.0f);
    spline.update(0.8f, 3.0f, 3.0f);
  }

  @Test
  public void modelRecord() {
    MultiDimensionSpline a = getMultiDimensionSpline();
    set(a);
    ModelRecord record = a.toModelRecord("","");

    MultiDimensionSpline b = new MultiDimensionSpline(record);
    assertEquals(0.40389338302461303, a.evaluate(3.0f, 3.0f), 0.0001);

    assertEquals(0.40389338302461303, b.evaluate(3.0f, 3.0f), 0.0001);
  }

  @Test
  public void testLInfinityNorm() {
    MultiDimensionSpline a = getMultiDimensionSpline();
    set(a);
    assertEquals(0.5676819, a.LInfinityNorm(), 0.001);
    a.update(-8.8f, 3.0f, 3.0f);
    assertEquals(2.2953262329101562, a.LInfinityNorm(), 0.001);
  }

  @Test
  public void testLInfinityCap() {
    MultiDimensionSpline a = getMultiDimensionSpline();
    set(a);
    a.LInfinityCap(0.8f);
    assertEquals(0.5676819, a.LInfinityNorm(), 0.001);
    a.LInfinityCap(0.5f);
    assertEquals(0.5, a.LInfinityNorm(), 0.001);
  }
}