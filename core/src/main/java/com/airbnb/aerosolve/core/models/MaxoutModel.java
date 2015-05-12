package com.airbnb.aerosolve.core.models;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;

import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.util.Util;
import com.airbnb.aerosolve.core.util.FloatVector;
import lombok.Getter;
import lombok.Setter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

// A 2 layer maxout unit that can represent functions using difference
// of piecewise linear convex functions.
// http://arxiv.org/abs/1302.4389
public class MaxoutModel extends AbstractModel {

  private static final long serialVersionUID = -849900702679383422L;

  @Getter @Setter
  private int numHidden;

  @Getter @Setter
  private Map<String, Map<String, WeightVector>> weightVector;

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

  public MaxoutModel() {
  }

  public void initForTraining(int numHidden) {
    this.numHidden = numHidden;
    weightVector = new HashMap<>();
    bias = new WeightVector(1.0f, numHidden, false);
    Map<String, WeightVector> special = new HashMap<>();
    weightVector.put("$SPECIAL", special);
    special.put("$BIAS", bias);
  }

  @Override
  public float scoreItem(FeatureVector combinedItem) {
    Map<String, Map<String, Double>> flatFeatures = Util.flattenFeature(combinedItem);
    return scoreFlatFeatures(flatFeatures);
  }

  @Override
  public float debugScoreItem(FeatureVector combinedItem,
                              StringBuilder builder) {
    Map<String, Map<String, Double>> flatFeatures = Util.flattenFeature(combinedItem);

    FloatVector response = getResponse(flatFeatures);
    FloatVector.MinMaxResult result = response.getMinMaxResult();

    float sum = result.maxValue - result.minValue;

    PriorityQueue<Map.Entry<String, Float>> scores =
        new PriorityQueue<>(100, new LinearModel.EntryComparator());

    float[] biasWt = bias.weights.getValues();
    float biasScore = biasWt[result.maxIndex] - biasWt[result.minIndex];
    scores.add(new AbstractMap.SimpleEntry<String, Float>(
        "bias = " + biasScore + " <br>\n",
        biasScore));

    for (Map.Entry<String, Map<String, Double>> featureFamily : flatFeatures.entrySet()) {
      Map<String, WeightVector> familyWeightMap = weightVector.get(featureFamily.getKey());
      if (familyWeightMap == null) continue;
      for (Map.Entry<String, Double> feature : featureFamily.getValue().entrySet()) {
        WeightVector weightVec = familyWeightMap.get(feature.getKey());
        if (weightVec == null) continue;
        float val = feature.getValue().floatValue();
        float[] wt = weightVec.weights.getValues();
        float p = wt[result.maxIndex] * weightVec.scale;
        float n = wt[result.minIndex] * weightVec.scale;
        float subscore = val * (p - n);
        String str = featureFamily.getKey() + ":" + feature.getKey() + "=" + val
            + " * (" + p + " - " + n + ") = " + subscore + "<br>\n";
        scores.add(new AbstractMap.SimpleEntry<String, Float>(str, subscore));
      }
    }
    builder.append("Top 15 scores ===>\n");
    if (!scores.isEmpty()) {
      int count = 0;
      float subsum = 0.0f;
      while (!scores.isEmpty()) {
        Map.Entry<String, Float> entry = scores.poll();
        builder.append(entry.getKey());
        float val = entry.getValue();
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
  public void addVector(String family, String feature, float scale) {
    Map<String, WeightVector> featFamily = weightVector.get(family);
    if (featFamily == null) {
      featFamily = new HashMap<>();
      weightVector.put(family, featFamily);
    }
    WeightVector vec = new WeightVector(scale, numHidden, true);
    featFamily.put(feature, vec);
  }

  // Updates the gradient
  public void update(float grad,
                     float learningRate,
                     float l1Reg,
                     float l2Reg,
                     float momentum,
                     FloatVector.MinMaxResult result,
                     Map<String, Map<String, Double>> flatFeatures) {
    for (Map.Entry<String, Map<String, Double>> featureFamily : flatFeatures.entrySet()) {
      Map<String, WeightVector> familyWeightMap = weightVector.get(featureFamily.getKey());
      if (familyWeightMap == null) continue;
      for (Map.Entry<String, Double> feature : featureFamily.getValue().entrySet()) {
        WeightVector weightVec = familyWeightMap.get(feature.getKey());
        if (weightVec == null) continue;
        float val = feature.getValue().floatValue() * weightVec.scale;
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

  public float scoreFlatFeatures(Map<String, Map<String, Double>> flatFeatures) {
    FloatVector response = getResponse(flatFeatures);
    FloatVector.MinMaxResult result = response.getMinMaxResult();

    return result.maxValue - result.minValue;
  }

  public FloatVector getResponse(Map<String, Map<String, Double>> flatFeatures) {
    FloatVector sum = new FloatVector(numHidden);
    for (Map.Entry<String, Map<String, Double>> entry : flatFeatures.entrySet()) {
      Map<String, WeightVector> family = weightVector.get(entry.getKey());
      if (family != null) {
        for (Map.Entry<String, Double> feature : entry.getValue().entrySet()) {
          WeightVector hidden = family.get(feature.getKey());
          if (hidden != null) {
            sum.multiplyAdd(feature.getValue().floatValue() * hidden.scale, hidden.weights);
          }
        }
      }
    }
    sum.add(bias.weights);
    return sum;
  }

  public void save(BufferedWriter writer) throws IOException {
    ModelHeader header = new ModelHeader();
    header.setModelType("maxout");
    header.setNumHidden(numHidden);
    long count = 0;
    for (Map.Entry<String, Map<String, WeightVector>> familyMap : weightVector.entrySet()) {
      for (Map.Entry<String, WeightVector> feature : familyMap.getValue().entrySet()) {
        count++;
      }
    }
    header.setNumRecords(count);
    ModelRecord headerRec = new ModelRecord();
    headerRec.setModelHeader(header);
    writer.write(Util.encode(headerRec));
    writer.newLine();
    for (Map.Entry<String, Map<String, WeightVector>> familyMap : weightVector.entrySet()) {
      for (Map.Entry<String, WeightVector> feature : familyMap.getValue().entrySet()) {
        ModelRecord record = new ModelRecord();
        record.setFeatureFamily(familyMap.getKey());
        record.setFeatureName(feature.getKey());
        ArrayList<Double> arrayList = new ArrayList<Double>();
        for (int i = 0; i < feature.getValue().weights.values.length; i++) {
          arrayList.add((double) feature.getValue().weights.values[i]);
        }
        record.setWeightVector(arrayList);
        record.setScale(feature.getValue().scale);
        writer.write(Util.encode(record));
        writer.newLine();
      }
    }
    writer.flush();
  }

  @Override
  protected void loadInternal(ModelHeader header, BufferedReader reader) throws IOException {
    long rows = header.getNumRecords();
    numHidden = header.getNumHidden();
    weightVector = new HashMap<>();
    for (long i = 0; i < rows; i++) {
      String line = reader.readLine();
      ModelRecord record = Util.decodeModel(line);
      String family = record.getFeatureFamily();
      String name = record.getFeatureName();
      Map<String, WeightVector> inner = weightVector.get(family);
      if (inner == null) {
        inner = new HashMap<>();
        weightVector.put(family, inner);
      }
      WeightVector vec = new WeightVector();
      vec.scale = (float) record.getScale();
      vec.weights = new FloatVector(numHidden);
      for (int j = 0; j < numHidden; j++) {
        vec.weights.values[j] = record.getWeightVector().get(j).floatValue();
      }
      inner.put(name, vec);
    }
    Map<String, WeightVector> special = weightVector.get("$SPECIAL");
    assert(special != null);
    bias = special.get("$BIAS");
    assert(bias != null);
  }
}