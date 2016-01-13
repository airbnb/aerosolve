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
import com.airbnb.aerosolve.core.MulticlassScoringOptions;
import com.airbnb.aerosolve.core.MulticlassScoringResult;
import com.airbnb.aerosolve.core.util.Util;
import com.airbnb.aerosolve.core.util.FloatVector;

import lombok.Getter;
import lombok.Setter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

// A full rank linear model that supports multi-class classificaiton.
// The class vector Y = W' X where X is the feature vector.
public class FullRankLinearModel extends AbstractModel {

  private static final long serialVersionUID = -849900702679383420L;

  @Getter @Setter
  private Map<String, Map<String, FloatVector>> weightVector;
  
  @Getter @Setter
  private ArrayList<LabelDictionaryEntry> labelDictionary;

  public FullRankLinearModel() {
  }

  // In the binary case this is just the score for class 0.
  @Override
  public float scoreItem(FeatureVector combinedItem) {
    Map<String, Map<String, Double>> flatFeatures = Util.flattenFeature(combinedItem);
    return 0.0f;
  }

  @Override
  public float debugScoreItem(FeatureVector combinedItem,
                              StringBuilder builder) {
    // TODO(hector_yee) : implement debug.
    return scoreItem(combinedItem);
  }

  @Override
  public List<DebugScoreRecord> debugScoreComponents(FeatureVector combinedItem) {
    // (TODO) implement debugScoreComponents
    List<DebugScoreRecord> scoreRecordsList = new ArrayList<>();
    return scoreRecordsList;
  }
  
  public ArrayList<MulticlassScoringResult> scoreItemMulticlass(FeatureVector combinedItem, MulticlassScoringOptions options) {
    ArrayList<MulticlassScoringResult> results = new ArrayList<>();
    
    return results;
  }

  public void save(BufferedWriter writer) throws IOException {
    ModelHeader header = new ModelHeader();
    header.setModelType("full_rank_linear");
    long count = 0;
    for (Map.Entry<String, Map<String, FloatVector>> familyMap : weightVector.entrySet()) {
      for (Map.Entry<String, FloatVector> feature : familyMap.getValue().entrySet()) {
        count++;
      }
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
      FloatVector vec = new FloatVector();
      for (int j = 0; j < record.getWeightVector().size(); j++) {
        vec.values[j] = record.getWeightVector().get(j).floatValue();
      }
      inner.put(name, vec);
    }
  }
}