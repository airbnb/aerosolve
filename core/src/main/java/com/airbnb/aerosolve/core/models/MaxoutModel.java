package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.perf.Feature;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.perf.FeatureValue;
import com.airbnb.aerosolve.core.util.FloatVector;
import com.airbnb.aerosolve.core.util.Util;
import it.unimi.dsi.fastutil.objects.Reference2ObjectMap;
import it.unimi.dsi.fastutil.objects.Reference2ObjectOpenHashMap;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import lombok.Getter;
import lombok.Setter;

// A 2 layer maxout unit that can represent functions using difference
// of piecewise linear convex functions.
// http://arxiv.org/abs/1302.4389
public class MaxoutModel extends AbstractModel {

  private static final long serialVersionUID = -849900702679383422L;

  @Getter @Setter
  private int numHidden;

  @Getter @Setter
  private Reference2ObjectMap<Feature, WeightVector> weightVector;

  private WeightVector bias;

  public static class WeightVector implements Serializable {
    private static final long serialVersionUID = -2698305146144718441L;
    WeightVector() {

    }
    WeightVector(float scale, int dim, boolean gaussian) {
      this.scale = scale;
      if (gaussian) {
        weights = FloatVector.getGaussianVector(dim);
      } else {
        weights = new FloatVector(dim);
      }
      ssg = new FloatVector(dim);
      prevStep = new FloatVector(dim);
    }
    public FloatVector weights;
    // Sum of squared gradients.
    public FloatVector ssg;
    // Previous step
    public FloatVector prevStep;
    public float scale;
  }

  public MaxoutModel(FeatureRegistry registry) {
    super(registry);
  }

  public void initForTraining(int numHidden) {
    this.numHidden = numHidden;
    weightVector = new Reference2ObjectOpenHashMap<>();
    bias = new WeightVector(1.0f, numHidden, false);
    Feature specialBias = registry.feature("$SPECIAL", "$BIAS");
    weightVector.put(specialBias, bias);
  }

  @Override
  public double scoreItem(FeatureVector combinedItem) {
    FloatVector response = getResponse(combinedItem);
    FloatVector.MinMaxResult result = response.getMinMaxResult();

    return result.maxValue - result.minValue;
  }

  @Override
  public double debugScoreItem(FeatureVector combinedItem,
                              StringBuilder builder) {

    FloatVector response = getResponse(combinedItem);
    FloatVector.MinMaxResult result = response.getMinMaxResult();

    double sum = result.maxValue - result.minValue;

    PriorityQueue<Map.Entry<String, Double>> scores =
        new PriorityQueue<>(100, new LinearModel.EntryComparator());

    float[] biasWt = bias.weights.getValues();
    double biasScore = biasWt[result.maxIndex] - biasWt[result.minIndex];
    scores.add(new AbstractMap.SimpleEntry<>(
        "bias = " + biasScore + " <br>\n",
        biasScore));

    for (FeatureValue value : combinedItem) {
      WeightVector weightVec = weightVector.get(value.feature());
      if (weightVec == null) continue;
      Feature feature = value.feature();
      double val = value.getDoubleValue();
      float[] wt = weightVec.weights.getValues();
      float p = wt[result.maxIndex] * weightVec.scale;
      float n = wt[result.minIndex] * weightVec.scale;
      double subscore = val * (p - n);
      String str = feature.family().name() + ":" + feature.name() + "=" + val
          + " * (" + p + " - " + n + ") = " + subscore + "<br>\n";
      scores.add(new AbstractMap.SimpleEntry<>(str, subscore));
    }
    builder.append("Top 15 scores ===>\n");
    if (!scores.isEmpty()) {
      int count = 0;
      float subsum = 0.0f;
      while (!scores.isEmpty()) {
        Map.Entry<String, Double> entry = scores.poll();
        builder.append(entry.getKey());
        double val = entry.getValue();
        subsum += val;
        count = count + 1;
        if (count >= 15) {
          builder.append("Leftover = " + (sum - subsum) + '\n');
          break;
        }
      }
    }
    builder.append("Total = " + sum + '\n');

    return sum;
  }

  @Override
  public List<DebugScoreRecord> debugScoreComponents(FeatureVector combinedItem) {
    // (TODO) implement debugScoreComponents
    List<DebugScoreRecord> scoreRecordsList = new ArrayList<>();
    return scoreRecordsList;
  }

  // Adds a new vector with a specified scale.
  public void addVector(String family, String featureName, float scale) {
    Feature feature = registry.feature(family, featureName);
    WeightVector vec = new WeightVector(scale, numHidden, true);
    weightVector.put(feature, vec);
  }

  // Updates the gradient
  public void update(float grad,
                     float learningRate,
                     float l1Reg,
                     float l2Reg,
                     float momentum,
                     FloatVector.MinMaxResult result,
                     FeatureVector vector) {
    for (FeatureValue value : vector) {
      WeightVector weightVec = weightVector.get(value.feature());
      if (weightVec == null) continue;
      float val = (float) value.getDoubleValue() * weightVec.scale;
      updateWeightVector(result.minIndex,
                         result.maxIndex,
                         val,
                         grad,
                         learningRate,
                         l1Reg,
                         l2Reg,
                         momentum,
                         weightVec);
    }
    updateWeightVector(result.minIndex,
                       result.maxIndex,
                       1.0f,
                       grad,
                       learningRate,
                       l1Reg,
                       l2Reg,
                       momentum,
                       bias);
  }

  private void updateWeightVector(int minIndex,
                                  int maxIndex,
                                  float val,
                                  float grad,
                                  float learningRate,
                                  float l1Reg,
                                  float l2Reg,
                                  float momentum,
                                  WeightVector weightVec) {
    float[] ssg = weightVec.ssg.getValues();
    float[] wt = weightVec.weights.getValues();
    float[] prev = weightVec.prevStep.getValues();
    ssg[maxIndex] += grad * grad;
    ssg[minIndex] += grad * grad;
    float newMax = fobosUpdate(wt[maxIndex],
                               grad * val,
                               learningRate,
                               l1Reg, l2Reg, ssg[maxIndex]);
    float stepMax = newMax - wt[maxIndex];
    if (newMax == 0.0f) {
      wt[maxIndex] = 0.0f;
      prev[maxIndex] = 0.0f;
    } else {
      wt[maxIndex] = wt[maxIndex] + stepMax + momentum * prev[maxIndex];
      prev[maxIndex] = stepMax;
    }
    float newMin = fobosUpdate(wt[minIndex],
                               -grad * val,
                               learningRate,
                               l1Reg, l2Reg, ssg[minIndex]);
    float stepMin = newMin - wt[minIndex];
    if (newMin == 0.0f) {
      wt[minIndex] = 0.0f;
      prev[minIndex] = 0.0f;
    } else {
      wt[minIndex] = wt[minIndex] + stepMin + momentum * prev[minIndex];
      prev[minIndex] = stepMin;
    }
  }

  public FloatVector getResponse(FeatureVector vector) {
    FloatVector sum = new FloatVector(numHidden);
    for (FeatureValue value : vector) {
      WeightVector hidden = weightVector.get(value.feature());
      if (hidden != null) {
        sum.multiplyAdd(value.getDoubleValue() * hidden.scale, hidden.weights);
      }
    }
    sum.add(bias.weights);
    return sum;
  }

  public void save(BufferedWriter writer) throws IOException {
    ModelHeader header = new ModelHeader();
    header.setModelType("maxout");
    header.setNumHidden(numHidden);
    header.setNumRecords(weightVector.size());
    ModelRecord headerRec = new ModelRecord();
    headerRec.setModelHeader(header);
    writer.write(Util.encode(headerRec));
    writer.newLine();
    for (Map.Entry<Feature, WeightVector> entry : weightVector.entrySet()) {
      ModelRecord record = new ModelRecord();
      Feature feature = entry.getKey();
      record.setFeatureFamily(feature.family().name());
      record.setFeatureName(feature.name());
      ArrayList<Double> arrayList = new ArrayList<Double>();
      for (int i = 0; i < entry.getValue().weights.values.length; i++) {
        arrayList.add((double) entry.getValue().weights.values[i]);
      }
      record.setWeightVector(arrayList);
      record.setScale(entry.getValue().scale);
      writer.write(Util.encode(record));
      writer.newLine();
    }
    writer.flush();
  }

  @Override
  protected void loadInternal(ModelHeader header, BufferedReader reader) throws IOException {
    long rows = header.getNumRecords();
    numHidden = header.getNumHidden();
    weightVector = new Reference2ObjectOpenHashMap<>();
    for (long i = 0; i < rows; i++) {
      String line = reader.readLine();
      ModelRecord record = Util.decodeModel(line);
      String family = record.getFeatureFamily();
      String name = record.getFeatureName();
      Feature feature = registry.feature(family, name);
      WeightVector vec = new WeightVector();
      vec.scale = (float) record.getScale();
      vec.weights = new FloatVector(numHidden);
      for (int j = 0; j < numHidden; j++) {
        vec.weights.values[j] = record.getWeightVector().get(j).floatValue();
      }
      weightVector.put(feature, vec);
    }
    assert(weightVector.containsKey(registry.family("$SPECIAL").feature("$BIAS")));
  }
}