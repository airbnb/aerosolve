package com.airbnb.aerosolve.core.function;

import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.NDTreeNode;
import com.airbnb.aerosolve.core.models.NDTreeModel;
import lombok.extern.slf4j.Slf4j;

import java.util.*;

@Slf4j
public class MultiDimensionSpline implements Function {
  private static final long serialVersionUID = 5166347177557769302L;
  private final NDTreeModel ndTreeModel;
  // NDTree leaf maps to spline point
  private final Map<Integer, List<MultiDimensionPoint>> weights;
  private final List<MultiDimensionPoint> points;

  public MultiDimensionSpline(NDTreeModel ndTreeModel) {
    this.ndTreeModel = ndTreeModel;
    Map<List<Float>, MultiDimensionPoint> pointsMap = new HashMap<>();
    weights = new HashMap<>();

    NDTreeNode[] nodes = ndTreeModel.getNodes();
    for (int i = 0; i < nodes.length; i++) {
      NDTreeNode node = nodes[i];
      if (node.getCoordinateIndex() == NDTreeModel.LEAF) {
        List<MultiDimensionPoint> list = MultiDimensionPoint.getCombinationWithoutDuplication(
            node.getMin(), node.getMax(), pointsMap);
        if (list != null && !list.isEmpty()) {
          weights.put(i, list);
        } else {
          log.info("leaf node return no MultiDimensionPoint {}", node);
        }
      }
    }
    points = new ArrayList<>(pointsMap.values());
  }

  public MultiDimensionSpline(NDTreeModel ndTreeModel, Map<List<Double>, Double> weights) {
    this(ndTreeModel);
    updateWeights(weights);
  }

  @Override // it doesn't need numBins just like linear
  public Function aggregate(Iterable<Function> functions, float scale, int numBins) {
    // functions size == 1/scale
    int length = points.size();
    float[] aggWeights = new float[length];
    for (Function fun: functions) {
      MultiDimensionSpline spline = (MultiDimensionSpline) fun;

      for (int i = 0; i < length; i++) {
        aggWeights[i] += scale * spline.points.get(i).getWeight();
      }
    }
    for (int i = 0; i < length; i++) {
      points.get(i).setWeight(aggWeights[i]);
    }
    return this;
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
    ModelRecord record = new ModelRecord();
    record.setFunctionForm(FunctionForm.MULTI_SPINE);
    record.setFeatureFamily(featureFamily);
    record.setWeightMap(getWeightsFromList());
    record.setNdtreeModel(Arrays.asList(ndTreeModel.getNodes()));
    return record;
  }

  private Map<List<Double>, Double> getWeightsFromList() {
    Map<List<Double>, Double> weights = new HashMap<>();
    for (MultiDimensionPoint p: points) {
      weights.put(toDouble(p.getCoordinates()), p.getWeight());
    }
    return weights;
  }

  private void updateWeights(Map<List<Double>, Double> map) {
    Map<List<Float>, Double> weights = new HashMap<>();
    for (Map.Entry<List<Double>, Double> entry : map.entrySet()) {

      weights.put(toFloat(entry.getKey()), entry.getValue());
    }
    for (MultiDimensionPoint p : points) {
      p.setWeight(weights.get(p.getCoordinates()));
    }
  }


  public static List<Double> toDouble(List<Float> list) {
    List<Double> r = new ArrayList<>(list.size());
    for (Float f: list) {
      r.add(f.doubleValue());
    }
    return r;
  }

  public static List<Float> toFloat(List<Double> list) {
    List<Float> r = new ArrayList<>(list.size());
    for (Double f: list) {
      r.add(f.floatValue());
    }
    return r;
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
