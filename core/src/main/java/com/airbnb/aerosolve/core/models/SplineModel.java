package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.function.Spline;
import com.airbnb.aerosolve.core.util.Util;
import lombok.Getter;
import lombok.Setter;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;

// A linear piecewise spline based model with a spline per feature.
// See http://en.wikipedia.org/wiki/Generalized_additive_model
public class SplineModel extends AbstractModel {

  private static final long serialVersionUID = 3651061358422885397L;
  @Getter @Setter
  private int numBins;

  @Getter @Setter
  private Map<String, Map<String, WeightSpline>> weightSpline;

  @Getter @Setter
  // Cap on the L_infinity norm of the spline. Defaults to 0 which is no cap.
  private float splineNormCap;

  public static class WeightSpline implements Serializable {
    private static final long serialVersionUID = -2884260218927875694L;

    public WeightSpline() {
    }

    public WeightSpline(float minVal, float maxVal, int numBins) {
      splineWeights = new float[numBins];
      spline = new Spline(minVal, maxVal, splineWeights);
    }
    
    public void resample(int newBins) {
      spline.resample(newBins);
      splineWeights = spline.getWeights();
    }
    public Spline spline;
    public float[] splineWeights;
    public float L1Norm() {
      float sum = 0.0f;
      for (int i = 0; i < splineWeights.length; i++) {
        sum += Math.abs(splineWeights[i]);
      }
      return sum;
    }
    public float LInfinityNorm() {
      float best = 0.0f;
      for (int i = 0; i < splineWeights.length; i++) {
        best = Math.max(best, Math.abs(splineWeights[i]));
      }
      return best;
    }
    public void LInfinityCap(float cap) {
      if (cap <= 0.0f) return;
      float currentNorm = this.LInfinityNorm();
      if (currentNorm > cap) {
        float scale = cap / currentNorm;
        for (int i = 0; i < splineWeights.length; i++) {
          splineWeights[i] *= scale;
        }
      }
    }
  }

  public SplineModel() {
  }

  public void initForTraining(int numBins) {
    this.numBins = numBins;
    weightSpline = new HashMap<>();
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

    float sum = 0.0f;

    PriorityQueue<Map.Entry<String, Float>> scores =
        new PriorityQueue<>(100, new LinearModel.EntryComparator());

    for (Map.Entry<String, Map<String, Double>> featureFamily : flatFeatures.entrySet()) {
      Map<String, WeightSpline> familyWeightMap = weightSpline.get(featureFamily.getKey());
      if (familyWeightMap == null) continue;
      for (Map.Entry<String, Double> feature : featureFamily.getValue().entrySet()) {
        WeightSpline ws = familyWeightMap.get(feature.getKey());
        if (ws == null) continue;
        float val = feature.getValue().floatValue();
        float subscore = ws.spline.evaluate(val);
        sum += subscore;
        String str = featureFamily.getKey() + ":" + feature.getKey() + "=" + val
            + " = " + subscore + "<br>\n";
        scores.add(new AbstractMap.SimpleEntry<String, Float>(str, subscore));
      }
    }
    final int MAX_COUNT = 100;
    builder.append("Top scores ===>\n");
    if (!scores.isEmpty()) {
      int count = 0;
      float subsum = 0.0f;
      while (!scores.isEmpty()) {
        Map.Entry<String, Float> entry = scores.poll();
        builder.append(entry.getKey());
        float val = entry.getValue();
        subsum += val;
        count = count + 1;
        if (count >= MAX_COUNT) {
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
    Map<String, Map<String, Double>> flatFeatures = Util.flattenFeature(combinedItem);
    List<DebugScoreRecord> scoreRecordsList = new ArrayList<>();

    for (Map.Entry<String, Map<String, Double>> featureFamily : flatFeatures.entrySet()) {
      Map<String, WeightSpline> familyWeightMap = weightSpline.get(featureFamily.getKey());
      if (familyWeightMap == null) continue;
      for (Map.Entry<String, Double> feature : featureFamily.getValue().entrySet()) {
        WeightSpline ws = familyWeightMap.get(feature.getKey());
        if (ws == null) continue;
        float val = feature.getValue().floatValue();
        float weight = ws.spline.evaluate(val);
        DebugScoreRecord record = new DebugScoreRecord();
        record.setFeatureFamily(featureFamily.getKey());
        record.setFeatureName(feature.getKey());
        record.setFeatureValue(val);
        record.setFeatureWeight(weight);
        scoreRecordsList.add(record);
      }
    }
    return scoreRecordsList;
  }

  // Updates the gradient
  public void update(float grad,
                     float learningRate,
                     Map<String, Map<String, Double>> flatFeatures) {
    for (Map.Entry<String, Map<String, Double>> featureFamily : flatFeatures.entrySet()) {
      Map<String, WeightSpline> familyWeightMap = weightSpline.get(featureFamily.getKey());
      if (familyWeightMap == null) continue;
      for (Map.Entry<String, Double> feature : featureFamily.getValue().entrySet()) {
        WeightSpline ws = familyWeightMap.get(feature.getKey());
        if (ws == null) continue;
        float val = feature.getValue().floatValue();
        updateWeightSpline(val, grad, learningRate,ws);
      }
    }
  }
  
  @Override
  public void onlineUpdate(float grad, float learningRate, Map<String, Map<String, Double>> flatFeatures) {
    update(grad, learningRate, flatFeatures);
  }

  // Adds a new spline
  public void addSpline(String family, String feature, float minVal, float maxVal, Boolean overwrite) {
    // if overwrite=true, we overwrite an existing spline, otherwise we don't modify an existing spline
    Map<String, WeightSpline> featFamily = weightSpline.get(family);
    if (featFamily == null) {
      featFamily = new HashMap<>();
      weightSpline.put(family, featFamily);
    }

    if (overwrite || !featFamily.containsKey(feature)) {
      if (maxVal <= minVal) {
        maxVal = minVal + 1.0f;
      }
      WeightSpline ws = new WeightSpline(minVal, maxVal, numBins);
      featFamily.put(feature, ws);
    }
  }

  private void updateWeightSpline(float val,
                                  float grad,
                                  float learningRate,
                                  WeightSpline ws) {
    ws.spline.update(-grad * learningRate, val);
    ws.LInfinityCap(splineNormCap);
  }

  public float scoreFlatFeatures(Map<String, Map<String, Double>> flatFeatures) {
    float sum = 0.0f;

    for (Map.Entry<String, Map<String, Double>> featureFamily : flatFeatures.entrySet()) {
      Map<String, WeightSpline> familyWeightMap = weightSpline.get(featureFamily.getKey());
      if (familyWeightMap == null)
        continue;
      for (Map.Entry<String, Double> feature : featureFamily.getValue().entrySet()) {
        WeightSpline ws = familyWeightMap.get(feature.getKey());
        if (ws == null)
          continue;
        float val = feature.getValue().floatValue();
        sum += ws.spline.evaluate(val);
      }
    }
    return sum;
  }

  @Override
  public void save(BufferedWriter writer) throws IOException {
    ModelHeader header = new ModelHeader();
    header.setModelType("spline");
    header.setNumHidden(numBins);
    header.setSlope(slope);
    header.setOffset(offset);
    long count = 0;
    for (Map.Entry<String, Map<String, WeightSpline>> familyMap : weightSpline.entrySet()) {
      for (Map.Entry<String, WeightSpline> feature : familyMap.getValue().entrySet()) {
        count++;
      }
    }
    header.setNumRecords(count);
    ModelRecord headerRec = new ModelRecord();
    headerRec.setModelHeader(header);
    writer.write(Util.encode(headerRec));
    writer.newLine();
    for (Map.Entry<String, Map<String, WeightSpline>> familyMap : weightSpline.entrySet()) {
      for (Map.Entry<String, WeightSpline> feature : familyMap.getValue().entrySet()) {
        ModelRecord record = new ModelRecord();
        record.setFeatureFamily(familyMap.getKey());
        record.setFeatureName(feature.getKey());
        ArrayList<Double> arrayList = new ArrayList<Double>();
        for (int i = 0; i < feature.getValue().splineWeights.length; i++) {
          arrayList.add((double) feature.getValue().splineWeights[i]);
        }
        record.setWeightVector(arrayList);
        record.setMinVal(feature.getValue().spline.getMinVal());
        record.setMaxVal(feature.getValue().spline.getMaxVal());
        writer.write(Util.encode(record));
        writer.newLine();
      }
    }
    writer.flush();
  }

  @Override
  protected void loadInternal(ModelHeader header, BufferedReader reader) throws IOException {
    long rows = header.getNumRecords();
    numBins = header.getNumHidden();
    slope = header.getSlope();
    offset = header.getOffset();
    weightSpline = new HashMap<>();

    for (long i = 0; i < rows; i++) {
      String line = reader.readLine();
      ModelRecord record = Util.decodeModel(line);
      String family = record.getFeatureFamily();
      String name = record.getFeatureName();
      Map<String, WeightSpline> inner = weightSpline.get(family);
      if (inner == null) {
        inner = new HashMap<>();
        weightSpline.put(family, inner);
      }
      float minVal = (float) record.getMinVal();
      float maxVal = (float) record.getMaxVal();
      WeightSpline vec = new WeightSpline(minVal, maxVal, numBins);
      for (int j = 0; j < numBins; j++) {
        vec.splineWeights[j] = record.getWeightVector().get(j).floatValue();
      }
      inner.put(name, vec);
    }
  }
}
