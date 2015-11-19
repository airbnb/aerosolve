package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.util.AbstractFunction;
import com.airbnb.aerosolve.core.util.Linear;
import com.airbnb.aerosolve.core.util.Polynomial;
import com.airbnb.aerosolve.core.util.Util;
import com.airbnb.aerosolve.core.util.Spline;
import lombok.Getter;
import lombok.Setter;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.AbstractMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.ArrayList;
import java.util.HashMap;

// A generalized additive model with a parametric function per feature.
// See http://en.wikipedia.org/wiki/Generalized_additive_model
public class AdditiveModel extends AbstractModel {
  @Getter @Setter
  private Map<String, Map<String, AbstractFunction>> weights;

  public AdditiveModel() {}

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
    // order by the absolute value
    PriorityQueue<Map.Entry<String, Float>> scores =
        new PriorityQueue<>(100, new LinearModel.EntryComparator());

    for (Map.Entry<String, Map<String, Double>> featureFamily : flatFeatures.entrySet()) {
      Map<String, AbstractFunction> familyWeightMap = weights.get(featureFamily.getKey());
      if (familyWeightMap == null)
        continue;
      for (Map.Entry<String, Double> feature : featureFamily.getValue().entrySet()) {
        AbstractFunction func = familyWeightMap.get(feature.getKey());
        if (func == null)
          continue;
        float val = feature.getValue().floatValue();
        float subscore = func.evaluate(val);
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
      Map<String, AbstractFunction> familyWeightMap = weights.get(featureFamily.getKey());
      if (familyWeightMap == null) continue;
      for (Map.Entry<String, Double> feature : featureFamily.getValue().entrySet()) {
        AbstractFunction func = familyWeightMap.get(feature.getKey());
        if (func == null) continue;
        float val = feature.getValue().floatValue();
        float weight = func.evaluate(val);
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

  @Override
  protected void loadInternal(ModelHeader header, BufferedReader reader) throws IOException {
    long rows = header.getNumRecords();
    slope = header.getSlope();
    offset = header.getOffset();
    weights = new HashMap<>();
    for (long i = 0; i < rows; i++) {
      String line = reader.readLine();
      ModelRecord record = Util.decodeModel(line);
      String family = record.getFeatureFamily();
      String name = record.getFeatureName();
      String funcForm = record.getFunctionForm();
      Map<String, AbstractFunction> inner = weights.get(family);
      if (inner == null) {
        inner = new HashMap<>();
        weights.put(family, inner);
      }
      if (funcForm.equalsIgnoreCase("Spline")) {
        inner.put(name, new Spline(record));
      } else if (funcForm.equalsIgnoreCase("Linear")) {
        inner.put(name, new Linear(record));
      }
    }
  }

  @Override
  public void save(BufferedWriter writer) throws IOException {
    ModelHeader header = new ModelHeader();
    header.setModelType("additive");
    header.setSlope(slope);
    header.setOffset(offset);
    long count = 0;
    for (Map.Entry<String, Map<String, AbstractFunction>> familyMap : weights.entrySet()) {
      for (Map.Entry<String, AbstractFunction> feature : familyMap.getValue().entrySet()) {
        count++;
      }
    }
    header.setNumRecords(count);
    ModelRecord headerRec = new ModelRecord();
    headerRec.setModelHeader(header);
    writer.write(Util.encode(headerRec));
    writer.newLine();
    for (Map.Entry<String, Map<String, AbstractFunction>> familyMap : weights.entrySet()) {
      String featureFamily = familyMap.getKey();
      for (Map.Entry<String, AbstractFunction> feature : familyMap.getValue().entrySet()) {
        AbstractFunction func = feature.getValue();
        String featureName = feature.getKey();
        writer.write(Util.encode(func.toModelRecord(featureFamily, featureName)));
        writer.newLine();
      }
    }
    writer.flush();
  }

  public float scoreFlatFeatures(Map<String, Map<String, Double>> flatFeatures) {
    float sum = 0.0f;

    for (Map.Entry<String, Map<String, Double>> featureFamily : flatFeatures.entrySet()) {
      Map<String, AbstractFunction> familyWeightMap = weights.get(featureFamily.getKey());
      if (familyWeightMap == null)
        continue;
      for (Map.Entry<String, Double> feature : featureFamily.getValue().entrySet()) {
        AbstractFunction func = familyWeightMap.get(feature.getKey());
        if (func == null)
          continue;
        float val = feature.getValue().floatValue();
        sum += func.evaluate(val);
      }
    }
    return sum;
  }
}
