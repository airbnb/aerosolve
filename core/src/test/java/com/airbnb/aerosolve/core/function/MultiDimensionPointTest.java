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
    List<Float> a = Arrays.asList((float)3.0, (float)-5.0, (float)-9.0);
    List<Float> a1 = Arrays.asList((float)3.0, (float)-5.0, (float)-9.0);
    List<Float> b = Arrays.asList((float)3.0, (float)-5.0, (float)10.0);

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
    List<List<Float>> r = Arrays.asList(
        Arrays.asList((float)3.0, (float)-5.0, (float)-9.0),
        Arrays.asList((float)13.0, (float)-5.0, (float)-9.0),
        Arrays.asList((float)3.0, (float)15.0, (float)-9.0),
        Arrays.asList((float)13.0, (float)15.0, (float)-9.0),
        Arrays.asList((float)3.0, (float)-5.0, (float)10.0),
        Arrays.asList((float)13.0, (float)-5.0, (float)10.0),
        Arrays.asList((float)3.0, (float)15.0, (float)10.0),
        Arrays.asList((float)13.0, (float)15.0, (float)10.0)
    );

    Map<List<Float>, MultiDimensionPoint> points = new HashMap<>();
    MultiDimensionPoint a = new MultiDimensionPoint(r.get(0));
    points.put(r.get(0), a);
    MultiDimensionPoint x = new MultiDimensionPoint(r.get(1));
    points.put(r.get(1), x);
    x.setWeight(0.5);
    List<Float> extra = Arrays.asList((float)5.0, (float)15.0, (float)11.0);
    points.put(extra, new MultiDimensionPoint(extra));

    List<MultiDimensionPoint> result = MultiDimensionPoint.getCombinationWithoutDuplication(min, max, points);
    assertEquals(8, result.size());
    assertEquals(9, points.size());

    assertEquals(a, result.get(0));
    assertTrue(a == result.get(0));
    MultiDimensionPoint y = result.get(1);
    assertEquals(x, y);
    assertTrue(x == y);
    assertEquals(0.5, y.getWeight(), 0.1);
  }

  @Test
  public void getCombination() throws Exception {
    List<Double> min = Arrays.asList(3.0, 5.0);
    List<Double> max = Arrays.asList(13.0, 15.0);
    List<List<Float>> keys = MultiDimensionPoint.getCombination(min, max);
    assertEquals(4, keys.size());
    List<List<Float>> r = Arrays.asList(
        Arrays.asList((float)3.0, (float)5.0),
        Arrays.asList((float)13.0, (float)5.0),
        Arrays.asList((float)3.0, (float)15.0),
        Arrays.asList((float)13.0, (float)15.0));
    assertEquals(keys, r);

    min = Arrays.asList(3.0, -5.0, -9.0);
    max = Arrays.asList(13.0, 15.0, 10.0);
    keys = MultiDimensionPoint.getCombination(min, max);
    assertEquals(8, keys.size());
    r = Arrays.asList(
        Arrays.asList((float)3.0, (float)-5.0, (float)-9.0),
        Arrays.asList((float)13.0, (float)-5.0, (float)-9.0),
        Arrays.asList((float)3.0, (float)15.0, (float)-9.0),
        Arrays.asList((float)13.0, (float)15.0, (float)-9.0),
        Arrays.asList((float)3.0, (float)-5.0, (float)10.0),
        Arrays.asList((float)13.0, (float)-5.0, (float)10.0),
        Arrays.asList((float)3.0, (float)15.0, (float)10.0),
        Arrays.asList((float)13.0, (float)15.0, (float)10.0)
    );
    assertEquals(keys, r);
  }

  @Test
  public void testZero() {
    float[] af = new float[]{0, 1, -2, (float) 3.4, (float)5.0, (float)-6.7,(float) 8.9};
    List<Float> al = Arrays.asList((float)0, (float)1, (float)-2, (float) 3.4, (float)5.0, (float)-6.7,(float) 8.9);
    List<Float> b = Arrays.asList((float)0, (float)1.0, (float) -2.0, (float)3.4, (float)5.0, (float)-6.7, (float)8.9);
    assertEquals(0, MultiDimensionPoint.euclideanDistance(af, al), 0d);
    assertEquals(0, MultiDimensionPoint.euclideanDistance(af, b), 0d);
  }

  @Test
  public void test() {
    float[] af = new float[]{(float)1.0, (float)-2.0, (float)3.0, (float)4.0};
    float[] bf = new float[]{(float)-5.0, (float)-6.0, (float)7.0, (float)8.0};
    List<Float> a = Arrays.asList((float)1.0, (float)-2.0, (float)3.0, (float)4.0);

    List<Float> b = Arrays.asList((float)-5.0, (float)-6.0, (float)7.0, (float)8.0);

    final double expected = Math.sqrt(84);
    assertEquals(expected, MultiDimensionPoint.euclideanDistance(af, b), 0.001d);
    assertEquals(expected, MultiDimensionPoint.euclideanDistance(bf, a), 0.001d);
  }
}