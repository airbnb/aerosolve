package com.airbnb.aerosolve.core.function;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/*
  represent a point in multi dimension space for Function
 */
public class MultiDimensionPoint implements Comparable<MultiDimensionPoint> {
  @Getter
  private List<Float> coordinates;
  @Getter @Setter
  private double weight;

  public MultiDimensionPoint(List<Float> coordinates) {
    this.coordinates = coordinates;
  }

  public void updateWeight(double delta) {
    weight += delta;
  }
  public void scaleWeight(float scale) {
    weight *= scale;
  }

  /*
    Generate combination coordinates from min and max list,
    Create new points if the coordinate is not in points map
    if it is in point map, reuse it.
    return all MultiDimensionPoint from the combination
    TODO FIX IT points is Float and min/max is Double, should be same, but due to
    all other Function and models use float. so points is Float.
    and thrift is use Double so it is List<Double>
   */
  public static List<MultiDimensionPoint> getCombinationWithoutDuplication(
      List<Double> min, List<Double> max, Map<List<Float>, MultiDimensionPoint> points) {
    List<List<Float>> keys = getCombination(min, max);

    List<MultiDimensionPoint> result = new ArrayList<>();
    for (List<Float> key: keys) {
      MultiDimensionPoint p = points.get(key);
      if (p == null) {
        p = new MultiDimensionPoint(key);
        points.put(key, p);
      }
      result.add(p);
    }
    return result;
  }

  public static List<List<Float>> getCombination(List<Double> min, List<Double> max) {
    List<List<Float>> keys = new ArrayList<>();
    assert (min.size() == max.size());
    int coordinateSize = min.size();
    int keySize = 1 << coordinateSize;

    for (int i = 0; i < keySize; ++i) {
      int k = i;
      List<Float> r = new ArrayList<>();
      for (int j = 0; j < coordinateSize; ++j) {
        if ((k & 1) == 1) {
          r.add(max.get(j).floatValue());
        } else {
          r.add(min.get(j).floatValue());
        }
        k >>= 1;
      }
      keys.add(r);
    }
    return keys;
  }

  @Override
  public boolean equals(Object aThat){
    if (this == aThat) return true;
    if (!(aThat instanceof MultiDimensionPoint)) {
      return false;
    }
    MultiDimensionPoint point = (MultiDimensionPoint) aThat;

    return coordinates.equals(point.coordinates);
  }

  @Override
  public int hashCode(){
    return coordinates.hashCode();
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    for (Float d: coordinates) {
      sb.append(d);
      sb.append(" ");
    }
    sb.append(" w ");
    sb.append(weight);
    return sb.toString();
  }

  public float getDistance(float[] coordinates) {
    return euclideanDistance(coordinates, this.coordinates);
  }

  public static float euclideanDistance(float[] x, List<Float> y) {
    assert (x.length == y.size());
    double sum = 0;
    for (int i = 0; i < x.length; i++) {
      final double dp = x[i] - y.get(i);
      sum += dp * dp;
    }
    return (float) Math.sqrt(sum);
  }

  @Override
  public int compareTo(MultiDimensionPoint o) {
    final int BEFORE = -1;
    final int EQUAL = 0;
    final int AFTER = 1;
    if (this == o) return EQUAL;

    //primitive numbers follow this form
    if (this.weight < o.weight) return BEFORE;
    if (this.weight > o.weight) return AFTER;
    return EQUAL;
  }
}
