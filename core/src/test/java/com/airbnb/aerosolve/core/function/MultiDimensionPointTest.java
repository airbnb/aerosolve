package com.airbnb.aerosolve.core.function;

import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

public class MultiDimensionPointTest {
  @Test
  public void testEquals() throws Exception {
    List<Double> a = Arrays.asList(3.0, -5.0, -9.0);
    List<Double> a1 = Arrays.asList(3.0, -5.0, -9.0);
    List<Double> b = Arrays.asList(3.0, -5.0, 10.0);
    MultiDimensionPoint p1 = new MultiDimensionPoint(a);
    MultiDimensionPoint p2 = new MultiDimensionPoint(a1);
    assertEquals(p1.hashCode(), p2.hashCode());
    assertEquals(p1, p2);
    MultiDimensionPoint p3 = new MultiDimensionPoint(b);
    assertFalse(p1.hashCode() == p3.hashCode());
    assertFalse(p1.equals(p3));
  }

  @Test
  public void getPointFromNDTree() throws Exception {
    List<Double> min = Arrays.asList(3.0, -5.0, -9.0);
    List<Double> max = Arrays.asList(13.0, 15.0, 10.0);
    List<List<Double>> r = Arrays.asList(
        Arrays.asList(3.0, -5.0, -9.0),
        Arrays.asList(13.0, -5.0, -9.0),
        Arrays.asList(3.0, 15.0, -9.0),
        Arrays.asList(13.0, 15.0, -9.0),
        Arrays.asList(3.0, -5.0, 10.0),
        Arrays.asList(13.0, -5.0, 10.0),
        Arrays.asList(3.0, 15.0, 10.0),
        Arrays.asList(13.0, 15.0, 10.0)
    );

    Map<List<Double>, MultiDimensionPoint> points = new HashMap<>();
    MultiDimensionPoint a = new MultiDimensionPoint(r.get(0));
    points.put(r.get(0), a);
    MultiDimensionPoint x = new MultiDimensionPoint(r.get(1));
    points.put(r.get(1), x);
    x.setWeight(0.5);
    List<Double> extra = Arrays.asList(5.0, 15.0, 11.0);
    points.put(extra, new MultiDimensionPoint(extra));

    List<MultiDimensionPoint> result = MultiDimensionPoint.getCombinationWithoutDuplication(min, max, points);
    assertEquals(8, result.size());
    assertEquals(9, points.size());

    assertEquals(a, result.get(0));
    assertTrue(a == result.get(0));
    MultiDimensionPoint y = result.get(1);
    System.out.println(y);
    assertEquals(x, y);
    assertTrue(x == y);
    assertEquals(0.5, y.getWeight(), 0.1);
  }

  @Test
  public void getCombination() throws Exception {
    List<Double> min = Arrays.asList(3.0, 5.0);
    List<Double> max = Arrays.asList(13.0, 15.0);
    List<List<Double>> keys = MultiDimensionPoint.getCombination(min, max);
    assertEquals(4, keys.size());
    List<List<Double>> r = Arrays.asList(
        Arrays.asList(3.0, 5.0),
        Arrays.asList(13.0, 5.0),
        Arrays.asList(3.0, 15.0),
        Arrays.asList(13.0, 15.0));
    assertEquals(keys, r);

    min = Arrays.asList(3.0, -5.0, -9.0);
    max = Arrays.asList(13.0, 15.0, 10.0);
    keys = MultiDimensionPoint.getCombination(min, max);
    assertEquals(8, keys.size());
    r = Arrays.asList(
        Arrays.asList(3.0, -5.0, -9.0),
        Arrays.asList(13.0, -5.0, -9.0),
        Arrays.asList(3.0, 15.0, -9.0),
        Arrays.asList(13.0, 15.0, -9.0),
        Arrays.asList(3.0, -5.0, 10.0),
        Arrays.asList(13.0, -5.0, 10.0),
        Arrays.asList(3.0, 15.0, 10.0),
        Arrays.asList(13.0, 15.0, 10.0)
    );
    assertEquals(keys, r);

  }

}