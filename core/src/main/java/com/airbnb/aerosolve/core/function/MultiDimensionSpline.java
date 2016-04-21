package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.NDTreeNode;
import com.airbnb.aerosolve.core.models.NDTreeModel;
import lombok.extern.slf4j.Slf4j;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Slf4j
public class MultiDimensionSpline implements Function {
  private static final long serialVersionUID = 5166347177557769302L;
  private NDTreeModel ndTreeModel;
  // NDTree leaf maps to spline point
  private Map<Integer, List<MultiDimensionPoint>> weights;

  public MultiDimensionSpline(NDTreeModel ndTreeModel) {
    this.ndTreeModel = ndTreeModel;
    Map<List<Float>, MultiDimensionPoint> points = new HashMap<>();
    weights = new HashMap<>();

    NDTreeNode[] nodes = ndTreeModel.getNodes();
    for (int i = 0; i < nodes.length; i++) {
      NDTreeNode node = nodes[i];
      if (node.getCoordinateIndex() == NDTreeModel.LEAF) {
        List<MultiDimensionPoint> list = MultiDimensionPoint.getCombinationWithoutDuplication(
            node.getMin(), node.getMax(), points);
        if (list != null && !list.isEmpty()) {
          weights.put(i, list);
        } else {
          log.info("leaf node return no MultiDimensionPoint {}", node);
        }
      }
    }
  }

  @Override
  public Function aggregate(Iterable<Function> functions, float scale, int numBins) {
    return null;
  }

  @Override
  public float evaluate(float ... coordinates) {
    List<MultiDimensionPoint> list = getNearbyPoints(coordinates);
    double[] distance = new double[list.size()];
    double sum = 0;
    for (int i = 0; i < list.size(); i++) {
      MultiDimensionPoint point = list.get(i);
      distance[i] = point.getDistance(coordinates);
      sum += distance[i];
    }
    float score = 0;
    for (int i = 0; i < list.size(); i++) {
      MultiDimensionPoint point = list.get(i);
      score += point.getWeight() * (distance[i]/sum);
    }
    return score;
  }

  @Override
  public void update(float delta, float ... values) {
    List<MultiDimensionPoint> list = getNearbyPoints(values);
    double[] distance = new double[list.size()];
    double sum = 0;
    for (int i = 0; i < list.size(); i++) {
      MultiDimensionPoint point = list.get(i);
      distance[i] = point.getDistance(values);
      sum += distance[i];
    }
    for (int i = 0; i < list.size(); i++) {
      MultiDimensionPoint point = list.get(i);
      point.updateWeight(delta * (distance[i]/sum));
    }
  }

  @Override
  public ModelRecord toModelRecord(String featureFamily, String featureName) {
    return null;
  }

  @Override
  public void setPriors(float[] params) {

  }

  @Override
  public void LInfinityCap(float cap) {

  }

  @Override
  public float LInfinityNorm() {
    return 0;
  }

  private List<MultiDimensionPoint> getNearbyPoints(float ... coordinates) {
    int index = ndTreeModel.leaf(coordinates);
    assert (index != -1 && weights.containsKey(index));
    return weights.get(index);
  }
}
