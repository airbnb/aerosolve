package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.functions.Spline;
import com.airbnb.aerosolve.core.util.Util;
import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.features.FeatureValue;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
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

// A linear piecewise spline based model with a spline per feature.
// See http://en.wikipedia.org/wiki/Generalized_additive_model
/*
 @deprecated Use AdditiveModel
 */
@Deprecated
public class SplineModel extends AbstractModel {

  private static final long serialVersionUID = 3651061358422885397L;
  @Getter @Setter
  private int numBins;

  @Getter @Setter
  private Map<Feature, WeightSpline> weightSpline;

  @Getter @Setter
  // Cap on the L_infinity norm of the spline. Defaults to 0 which is no cap.
  private float splineNormCap;

  public static class WeightSpline implements Serializable {
    private static final long serialVersionUID = -2884260218927875694L;

    public WeightSpline() {
    }

    public WeightSpline(double minVal, double maxVal, int numBins) {
      splineWeights = new double[numBins];
      spline = new Spline(minVal, maxVal, splineWeights);
    }
    
    public void resample(int newBins) {
      spline.resample(newBins);
      splineWeights = spline.getWeights();
    }

    public Spline spline;

    public double[] splineWeights;

    public double L1Norm() {
      double sum = 0.0f;
      for (int i = 0; i < splineWeights.length; i++) {
        sum += Math.abs(splineWeights[i]);
      }
      return sum;
    }

    public double LInfinityNorm() {
      double best = 0.0f;
      for (int i = 0; i < splineWeights.length; i++) {
        best = Math.max(best, Math.abs(splineWeights[i]));
      }
      return best;
    }

    public void LInfinityCap(float cap) {
      if (cap <= 0.0f) return;
      double currentNorm = this.LInfinityNorm();
      if (currentNorm > cap) {
        double scale = cap / currentNorm;
        for (int i = 0; i < splineWeights.length; i++) {
          splineWeights[i] *= scale;
        }
      }
    }
  }

  public SplineModel(FeatureRegistry registry) {
    super(registry);
  }

  public void initForTraining(int numBins) {
    this.numBins = numBins;
    weightSpline = new Object2ObjectOpenHashMap<>();
  }

  @Override
  public double scoreItem(FeatureVector combinedItem) {
    return scoreItemInternal(combinedItem, null, null);
  }

  private double scoreItemInternal(FeatureVector combinedItem,
                                   PriorityQueue<Map.Entry<FeatureValue, Double>> scores,
                                   List<DebugScoreRecord> scoreRecordsList) {
    double sum = 0.0d;

    for (FeatureValue value : combinedItem) {
      WeightSpline ws = weightSpline.get(value.feature());
      if (ws == null)
        continue;
      double val = value.value();
      double subscore = ws.spline.evaluate(val);
      sum += subscore;
      if (scores != null) {
        scores.add(new AbstractMap.SimpleEntry<>(value, subscore));
      }
      if (scoreRecordsList != null) {
        DebugScoreRecord record = new DebugScoreRecord();
        record.setFeatureFamily(value.feature().family().name());
        record.setFeatureName(value.feature().name());
        record.setFeatureValue(val);
        record.setFeatureWeight(subscore);
        scoreRecordsList.add(record);
      }
    }
    return sum;
  }

  @Override
  public double debugScoreItem(FeatureVector combinedItem,
                              StringBuilder builder) {

    PriorityQueue<Map.Entry<FeatureValue, Double>> scores =
        new PriorityQueue<>(100, new LinearModel.EntryComparator());

    double sum = scoreItemInternal(combinedItem, scores, null);

    final int MAX_COUNT = 100;
    builder.append("Top scores ===>\n");
    if (!scores.isEmpty()) {
      int count = 0;
      float subsum = 0.0f;
      while (!scores.isEmpty()) {
        Map.Entry<FeatureValue, Double> entry = scores.poll();
        Feature feature = entry.getKey().feature();
        double subscore = entry.getValue();
        String str = feature.family().name() + ":" + feature.name()
                     + "=" + entry.getKey().value()
                     + " = " + subscore + "<br>\n";
        builder.append(str);
        subsum += subscore;
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
    List<DebugScoreRecord> scoreRecordsList = new ArrayList<>();

    scoreItemInternal(combinedItem, null, scoreRecordsList);
    return scoreRecordsList;
  }

  // Updates the gradient
  public void update(double grad,
                     double learningRate,
                     FeatureVector vector) {
    for (FeatureValue value : vector) {
      WeightSpline ws = weightSpline.get(value.feature());
      if (ws == null)
        continue;
      double val = value.value();
      updateWeightSpline(val, grad, learningRate, ws);
    }
  }
  
  @Override
  public void onlineUpdate(double grad, double learningRate, FeatureVector vector) {
    update(grad, learningRate, vector);
  }

  // Adds a new spline
  public void addSpline(Feature feature, double minVal, double maxVal,
                        boolean overwrite) {
    // if overwrite=true, we overwrite an existing spline, otherwise we don't modify an existing spline
    if (overwrite || !weightSpline.containsKey(feature)) {
      if (maxVal <= minVal) {
        maxVal = minVal + 1.0f;
      }
      WeightSpline ws = new WeightSpline(minVal, maxVal, numBins);
      weightSpline.put(feature, ws);
    }
  }

  private void updateWeightSpline(double val,
                                  double grad,
                                  double learningRate,
                                  WeightSpline ws) {
    ws.spline.update(-grad * learningRate, val);
    ws.LInfinityCap(splineNormCap);
  }

  @Override
  public void save(BufferedWriter writer) throws IOException {
    ModelHeader header = new ModelHeader();
    header.setModelType("spline");
    header.setNumHidden(numBins);
    header.setSlope(slope);
    header.setOffset(offset);
    header.setNumRecords(weightSpline.size());
    ModelRecord headerRec = new ModelRecord();
    headerRec.setModelHeader(header);
    writer.write(Util.encode(headerRec));
    writer.newLine();
    for (Map.Entry<Feature, WeightSpline> entry : weightSpline.entrySet()) {
      ModelRecord record = new ModelRecord();
      record.setFeatureFamily(entry.getKey().family().name());
      record.setFeatureName(entry.getKey().name());
      ArrayList<Double> arrayList = new ArrayList<Double>();
      for (int i = 0; i < entry.getValue().splineWeights.length; i++) {
        arrayList.add(entry.getValue().splineWeights[i]);
      }
      record.setWeightVector(arrayList);
      record.setMinVal(entry.getValue().spline.getMinVal());
      record.setMaxVal(entry.getValue().spline.getMaxVal());
      writer.write(Util.encode(record));
      writer.newLine();
    }
    writer.flush();
  }

  @Override
  protected void loadInternal(ModelHeader header, BufferedReader reader) throws IOException {
    long rows = header.getNumRecords();
    numBins = header.getNumHidden();
    slope = header.getSlope();
    offset = header.getOffset();
    weightSpline = new Reference2ObjectOpenHashMap<>();

    for (long i = 0; i < rows; i++) {
      String line = reader.readLine();
      ModelRecord record = Util.decodeModel(line);
      String family = record.getFeatureFamily();
      String name = record.getFeatureName();
      double minVal = record.getMinVal();
      double maxVal = record.getMaxVal();
      WeightSpline vec = new WeightSpline(minVal, maxVal, numBins);
      for (int j = 0; j < numBins; j++) {
        vec.splineWeights[j] = record.getWeightVector().get(j).floatValue();
      }
      weightSpline.put(registry.feature(family, name), vec);
    }
  }
}
