package com.airbnb.aerosolve.core.function;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/*
  represent a point in multi dimension space for Function
 */
public class MultiDimensionPoint {
  private List<Double> coordinates;
  @Getter @Setter
  private double weight;

  public MultiDimensionPoint(List<Double> coordinates) {
    this.coordinates = coordinates;
  }

  /*
    Generate combination coordinates from min and max list,
    Create new points if the coordinate is not in points map
    if it is in point map, reuse it.
    return all MultiDimensionPoint from the combination
   */
  public static List<MultiDimensionPoint> getCombinationWithoutDuplication(
      List<Double> min, List<Double> max, Map<List<Double>, MultiDimensionPoint> points) {
    List<List<Double>> keys = getCombination(min, max);

    List<MultiDimensionPoint> result = new ArrayList<>();
    for (List<Double> key: keys) {
      MultiDimensionPoint p = points.get(key);
      if (p == null) {
        p = new MultiDimensionPoint(key);
        points.put(key, p);
      }
      result.add(p);
    }
    return result;
  }

  public static List<List<Double>> getCombination(List<Double> min, List<Double> max) {
    List<List<Double>> keys = new ArrayList<>();
    assert (min.size() == max.size());
    int coordinateSize = min.size();
    int keySize = 1 << coordinateSize;

    for (int i = 0; i < keySize; ++i) {
      int k = i;
      List<Double> r = new ArrayList<>();
      for (int j = 0; j < coordinateSize; ++j) {
        if ((k & 1) == 1) {
          r.add(max.get(j));
        } else {
          r.add(min.get(j));
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
    for (Double d: coordinates) {
      sb.append(d);
      sb.append(" ");
    }
    sb.append(" w ");
    sb.append(weight);
    return sb.toString();
  }
}
