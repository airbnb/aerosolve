package com.airbnb.aerosolve.core.models;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.*;

import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.LabelDictionaryEntry;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.MulticlassScoringResult;
import com.airbnb.aerosolve.core.util.Util;
import com.airbnb.aerosolve.core.util.FloatVector;

import lombok.Getter;
import lombok.Setter;

// A full rank linear model that supports multi-class classification.
// The class vector Y = W' X where X is the feature vector.
// It is full rank because the matrix W is num-features by num-labels in dimension.
// Use a low rank model if you want better generalization.
public class FullRankLinearModel extends AbstractModel {

  private static final long serialVersionUID = -849900702679383420L;

  @Getter @Setter
  private Map<String, Map<String, FloatVector>> weightVector;
  
  @Getter @Setter
  private ArrayList<LabelDictionaryEntry> labelDictionary;

  @Getter @Setter
  private Map<String, Integer> labelToIndex;

  public FullRankLinearModel() {
    weightVector = new HashMap<>();
    labelDictionary = new ArrayList<>();
  }

  // In the binary case this is just the score for class 0.
  // Ideally use a binary model for binary classification.
  @Override
  public float scoreItem(FeatureVector combinedItem) {
    // Not supported.
    assert(false);
    Map<String, Map<String, Double>> flatFeatures = Util.flattenFeature(combinedItem);
    FloatVector sum = scoreFlatFeature(flatFeatures);
    return sum.values[0];
  }

  @Override
  public float debugScoreItem(FeatureVector combinedItem,
                              StringBuilder builder) {
    // TODO(hector_yee) : implement debug.
    return scoreItem(combinedItem);
  }

  @Override
  public List<DebugScoreRecord> debugScoreComponents(FeatureVector combinedItem) {
    Map<String, Map<String, Double>> flatFeatures = Util.flattenFeature(combinedItem);
    List<DebugScoreRecord> scoreRecordsList = new ArrayList<>();
    int dim = labelDictionary.size();
    for (Map.Entry<String, Map<String, Double>> entry : flatFeatures.entrySet()) {
      String familyKey = entry.getKey();
      Map<String, FloatVector> family = weightVector.get(familyKey);
      if (family != null) {
        for (Map.Entry<String, Double> feature : entry.getValue().entrySet()) {
          String featureKey = feature.getKey();
          FloatVector featureWeights = family.get(featureKey);
          float val = feature.getValue().floatValue();
          if (featureWeights != null) {
            for (int i = 0; i < dim; i++) {
              DebugScoreRecord record = new DebugScoreRecord();
              record.setFeatureFamily(familyKey);
              record.setFeatureName(featureKey);
              record.setFeatureValue(val);
              record.setFeatureWeight(featureWeights.get(i));
              record.setLabel(labelDictionary.get(i).label);
              scoreRecordsList.add(record);
            }
          }
        }
      }
    }
    return scoreRecordsList;
  }
  
  public ArrayList<MulticlassScoringResult> scoreItemMulticlass(FeatureVector combinedItem) {
    ArrayList<MulticlassScoringResult> results = new ArrayList<>();
    Map<String, Map<String, Double>> flatFeatures = Util.flattenFeature(combinedItem);
    FloatVector sum = scoreFlatFeature(flatFeatures);
    
    for (int i = 0; i < labelDictionary.size(); i++) {
      MulticlassScoringResult result = new MulticlassScoringResult();
      result.setLabel(labelDictionary.get(i).getLabel());
      result.setScore(sum.values[i]);
      results.add(result);
    }

    return results;
  }

  public FloatVector scoreFlatFeature(Map<String, Map<String, Double>> flatFeatures) {
    int dim = labelDictionary.size();
    FloatVector sum = new FloatVector(dim);

    for (Map.Entry<String, Map<String, Double>> entry : flatFeatures.entrySet()) {
      Map<String, FloatVector> family = weightVector.get(entry.getKey());
      if (family != null) {
        for (Map.Entry<String, Double> feature : entry.getValue().entrySet()) {
          FloatVector vec = family.get(feature.getKey());
          if (vec != null) {
            sum.multiplyAdd(feature.getValue().floatValue(), vec);
          }
        }
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
    long count = 0;
    for (Map.Entry<String, Map<String, FloatVector>> familyMap : weightVector.entrySet()) {
      count += familyMap.getValue().entrySet().size();
    }
    header.setNumRecords(count);
    header.setLabelDictionary(labelDictionary);
    ModelRecord headerRec = new ModelRecord();
    headerRec.setModelHeader(header);
    writer.write(Util.encode(headerRec));
    writer.newLine();
    for (Map.Entry<String, Map<String, FloatVector>> familyMap : weightVector.entrySet()) {
      for (Map.Entry<String, FloatVector> feature : familyMap.getValue().entrySet()) {
        ModelRecord record = new ModelRecord();
        record.setFeatureFamily(familyMap.getKey());
        record.setFeatureName(feature.getKey());
        ArrayList<Double> arrayList = new ArrayList<Double>();
        for (int i = 0; i < feature.getValue().values.length; i++) {
          arrayList.add((double) feature.getValue().values[i]);
        }
        record.setWeightVector(arrayList);
        writer.write(Util.encode(record));
        writer.newLine();
      }
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
    weightVector = new HashMap<>();
    for (long i = 0; i < rows; i++) {
      String line = reader.readLine();
      ModelRecord record = Util.decodeModel(line);
      String family = record.getFeatureFamily();
      String name = record.getFeatureName();
      Map<String, FloatVector> inner = weightVector.get(family);
      if (inner == null) {
        inner = new HashMap<>();
        weightVector.put(family, inner);
      }
      FloatVector vec = new FloatVector(record.getWeightVector().size());
      for (int j = 0; j < record.getWeightVector().size(); j++) {
        vec.values[j] = record.getWeightVector().get(j).floatValue();
      }
      inner.put(name, vec);
    }
  }
}