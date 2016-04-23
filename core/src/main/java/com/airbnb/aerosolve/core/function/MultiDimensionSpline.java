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

  public MultiDimensionSpline(ModelRecord record) {
    this(new NDTreeModel(record.getNdtreeModel()), record.getWeightVector());
  }

  public MultiDimensionSpline(NDTreeModel ndTreeModel, List<Double> weights) {
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
    record.setFunctionForm(FunctionForm.MultiDimensionSpline);
    record.setFeatureFamily(featureFamily);
    record.setWeightVector(getWeightsFromList());
    record.setNdtreeModel(Arrays.asList(ndTreeModel.getNodes()));
    return record;
  }

  private List<Double> getWeightsFromList() {
    List<Double> weights = new ArrayList<>(points.size());
    for (MultiDimensionPoint p: points) {

      weights.add(p.getWeight());
    }
    return weights;
  }

  private void updateWeights(List<Double> weights) {
    assert (weights.size() == points.size());
    for (int i = 0; i < points.size(); i++) {
      MultiDimensionPoint p = points.get(i);
      p.setWeight(weights.get(i));
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

  @Override public void setPriors(float[] params) {
    assert (params.length == points.size());
    for (int i = 0; i < points.size(); i++) {
      MultiDimensionPoint p = points.get(i);
      p.setWeight(params[i]);
    }
  }

  @Override
  public void LInfinityCap(float cap) {
    if (cap <= 0.0f) return;
    float currentNorm = LInfinityNorm();
    if (currentNorm > cap) {
      float scale = cap / currentNorm;
      for (int i = 0; i < points.size(); i++) {
        points.get(i).scaleWeight(scale);
      }
    }
  }

  @Override
  public float LInfinityNorm() {
    return (float) Math.max(Collections.max(points).getWeight(),
        Math.abs(Collections.min(points).getWeight()));
  }

  private List<MultiDimensionPoint> getNearbyPoints(float ... coordinates) {
    int index = ndTreeModel.leaf(coordinates);
    assert (index != -1 && weights.containsKey(index));
    return weights.get(index);
  }

  @Override
  public void resample(int newBins) {
  }
}
