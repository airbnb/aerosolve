package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.LabelDictionaryEntry;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.MulticlassScoringResult;
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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.Getter;
import lombok.Setter;

// A full rank linear model that supports multi-class classification.
// The class vector Y = W' X where X is the feature vector.
// It is full rank because the matrix W is num-features by num-labels in dimension.
// Use a low rank model if you want better generalization.
public class FullRankLinearModel extends AbstractModel {

  private static final long serialVersionUID = -849900702679383420L;

  @Getter
  private Reference2ObjectMap<Feature, FloatVector> weightVector;
  
  @Getter @Setter
  private ArrayList<LabelDictionaryEntry> labelDictionary;

  @Getter @Setter
  private Map<String, Integer> labelToIndex;

  public FullRankLinearModel(FeatureRegistry registry) {
    super(registry);
    weightVector = new Reference2ObjectOpenHashMap<>();
    labelDictionary = new ArrayList<>();
  }

  // In the binary case this is just the score for class 0.
  // Ideally use a binary model for binary classification.
  @Override
  public double scoreItem(FeatureVector combinedItem) {
    // Not supported.
    assert(false);
    FloatVector sum = scoreFlatFeature(combinedItem);
    return sum.values[0];
  }

  @Override
  public double debugScoreItem(FeatureVector combinedItem,
                              StringBuilder builder) {
    // TODO(hector_yee) : implement debug.
    return scoreItem(combinedItem);
  }

  @Override
  public List<DebugScoreRecord> debugScoreComponents(FeatureVector combinedItem) {
    List<DebugScoreRecord> scoreRecordsList = new ArrayList<>();
    int dim = labelDictionary.size();
    for (FeatureValue value : combinedItem) {
      FloatVector featureWeights = weightVector.get(value.feature());
      if (featureWeights != null) {
        for (int i = 0; i < dim; i++) {
          DebugScoreRecord record = new DebugScoreRecord();
          record.setFeatureFamily(value.feature().family().name());
          record.setFeatureName(value.feature().name());
          record.setFeatureValue(value.getDoubleValue());
          record.setFeatureWeight(featureWeights.get(i));
          record.setLabel(labelDictionary.get(i).label);
          scoreRecordsList.add(record);
        }
      }
    }
    return scoreRecordsList;
  }
  
  public ArrayList<MulticlassScoringResult> scoreItemMulticlass(FeatureVector combinedItem) {
    ArrayList<MulticlassScoringResult> results = new ArrayList<>();
    FloatVector sum = scoreFlatFeature(combinedItem);
    
    for (int i = 0; i < labelDictionary.size(); i++) {
      MulticlassScoringResult result = new MulticlassScoringResult();
      result.setLabel(labelDictionary.get(i).getLabel());
      result.setScore(sum.values[i]);
      results.add(result);
    }

    return results;
  }

  public FloatVector scoreFlatFeature(FeatureVector vector) {
    int dim = labelDictionary.size();
    FloatVector sum = new FloatVector(dim);

    for (FeatureValue value : vector) {
      FloatVector vec = weightVector.get(value.feature());
      if (vec != null) {
        sum.multiplyAdd(value.getDoubleValue(), vec);
      }
    }
    return sum;
  }

  public void buildLabelToIndex() {
    labelToIndex = new HashMap<>();
    for (int i = 0; i < labelDictionary.size(); i++) {
      labelToIndex.put(labelDictionary.get(i).label, i);
    }
  }

  public void save(BufferedWriter writer) throws IOException {
    ModelHeader header = new ModelHeader();
    header.setModelType("full_rank_linear");
    header.setNumRecords(weightVector.size());
    header.setLabelDictionary(labelDictionary);
    ModelRecord headerRec = new ModelRecord();
    headerRec.setModelHeader(header);
    writer.write(Util.encode(headerRec));
    writer.newLine();
    for (Map.Entry<Feature, FloatVector> entry : weightVector.entrySet()) {
      ModelRecord record = new ModelRecord();
      record.setFeatureFamily(entry.getKey().family().name());
      record.setFeatureName(entry.getKey().name());
      ArrayList<Double> arrayList = new ArrayList<Double>();
      for (int i = 0; i < entry.getValue().values.length; i++) {
        arrayList.add((double) entry.getValue().values[i]);
      }
      record.setWeightVector(arrayList);
      writer.write(Util.encode(record));
      writer.newLine();
    }
    writer.flush();
  }

  @Override
  protected void loadInternal(ModelHeader header, BufferedReader reader) throws IOException {
    long rows = header.getNumRecords();
    labelDictionary = new ArrayList<>();
    for (LabelDictionaryEntry entry : header.getLabelDictionary()) {
      labelDictionary.add(entry);
    }
    buildLabelToIndex();
    weightVector = new Reference2ObjectOpenHashMap<>();
    for (long i = 0; i < rows; i++) {
      String line = reader.readLine();
      ModelRecord record = Util.decodeModel(line);
      String family = record.getFeatureFamily();
      String name = record.getFeatureName();
      Feature feature = registry.feature(family, name);
      FloatVector vec = new FloatVector(record.getWeightVector().size());
      for (int j = 0; j < record.getWeightVector().size(); j++) {
        vec.values[j] = record.getWeightVector().get(j).floatValue();
      }
      weightVector.put(feature, vec);
    }
  }
}