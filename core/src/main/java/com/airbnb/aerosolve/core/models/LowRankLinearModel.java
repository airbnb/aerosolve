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

// A low rank linear model that supports multi-class classification.
// The class vector y = W' * V * x where x is d-dim feature vector.
// Suppose we have Y different labels and the D is the dimension of the joint feature-label space
// V: D-by-d matrix, mapping from feature space to the joint embedding
// W: D-by-Y matrix, mapping from label space to the joint embedding
// Reference: Jason Weston et al. "WSABIE: Scaling Up To Large Vocabulary Image Annotation", IJCAI 2011.
public class LowRankLinearModel extends AbstractModel {

  static final long serialVersionUID = -8894096678183767660L;

  // featureWeightVector represents the projection from feature space to embedding
  // Map feature family name, feature name to a column in V
  // each FloatVector in the map is a D-dim vector
  @Getter
  @Setter
  private Map<String, Map<String, FloatVector>> featureWeightVector;

  // labelWeightVector represents the projection from label space to embedding
  // Map label to a row in W, each FloatVector in the map is a D-dim vector
  @Getter
  @Setter
  private Map<String, FloatVector> labelWeightVector;

  @Getter
  @Setter
  private ArrayList<LabelDictionaryEntry> labelDictionary;

  @Getter
  @Setter
  private Map<String, Integer> labelToIndex;

  // size of the embedding
  @Getter
  @Setter
  private int embeddingDimension;

  public LowRankLinearModel() {
    featureWeightVector = new HashMap<>();
    labelWeightVector = new HashMap<>();
    labelDictionary = new ArrayList<>();
  }

  // In the binary case this is just the score for class 0.
  // Ideally use a binary model for binary classification.
  @Override
  public float scoreItem(FeatureVector combinedItem) {
    // Not supported.
    assert (false);
    Map<String, Map<String, Double>> flatFeatures = Util.flattenFeature(combinedItem);
    FloatVector sum = scoreFlatFeature(flatFeatures);
    return sum.values[0];
  }

  @Override
  public float debugScoreItem(FeatureVector combinedItem,
                              StringBuilder builder) {
    // TODO(peng) : implement debug.
    return scoreItem(combinedItem);
  }


  @Override
  public List<DebugScoreRecord> debugScoreComponents(FeatureVector combinedItem) {
    // TODO(peng): implement debugScoreComponents
    List<DebugScoreRecord> scoreRecordsList = new ArrayList<>();
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
    FloatVector fvProjection = projectFeatureToEmbedding(flatFeatures);
    return projectEmbeddingToLabel(fvProjection);
  }

  public FloatVector projectFeatureToEmbedding(Map<String, Map<String, Double>> flatFeatures) {
    FloatVector fvProjection = new FloatVector(embeddingDimension);

    // compute the projection from feature space to D-dim joint space
    for (Map.Entry<String, Map<String, Double>> entry : flatFeatures.entrySet()) {
      Map<String, FloatVector> family = featureWeightVector.get(entry.getKey());
      if (family != null) {
        for (Map.Entry<String, Double> feature : entry.getValue().entrySet()) {
          FloatVector vec = family.get(feature.getKey());
          if (vec != null) {
            fvProjection.multiplyAdd(feature.getValue().floatValue(), vec);
          }
        }
      }
    }
    return fvProjection;
  }

  public FloatVector projectEmbeddingToLabel(FloatVector fvProjection) {
    int dim = labelDictionary.size();
    FloatVector sum = new FloatVector(dim);
    // compute the projection from D-dim joint space to label space
    for (int i = 0; i < dim; i++) {
      String labelKey = labelDictionary.get(i).getLabel();
      FloatVector labelVector = labelWeightVector.get(labelKey);
      if (labelVector != null) {
        float val = labelVector.dot(fvProjection);
        sum.set(i, val);
      }
    }
    return sum;
  }

  public void buildLabelToIndex() {
    labelToIndex = new HashMap<>();
    for (int i = 0; i < labelDictionary.size(); i++) {
      String labelKey = labelDictionary.get(i).label;
      labelToIndex.put(labelKey, i);
    }
  }

  public void save(BufferedWriter writer) throws IOException {
    ModelHeader header = new ModelHeader();
    header.setModelType("low_rank_linear");
    long count = 0;
    for (Map.Entry<String, Map<String, FloatVector>> familyMap : featureWeightVector.entrySet()) {
      count += familyMap.getValue().entrySet().size();
    }
    header.setNumRecords(count);
    header.setLabelDictionary(labelDictionary);
    Map<String, java.util.List<Double>> labelEmbedding = new HashMap<>();
    for (Map.Entry<String, FloatVector> labelRepresentation : labelWeightVector.entrySet()) {
      float[] values = labelRepresentation.getValue().getValues();

      ArrayList<Double> arrayList = new ArrayList<>();
      for (int i = 0; i < embeddingDimension; i++) {
        arrayList.add((double) values[i]);
      }
      labelEmbedding.put(labelRepresentation.getKey(), arrayList);
    }
    header.setLabelEmbedding(labelEmbedding);

    ModelRecord headerRec = new ModelRecord();
    headerRec.setModelHeader(header);
    writer.write(Util.encode(headerRec));
    writer.newLine();
    for (Map.Entry<String, Map<String, FloatVector>> familyMap : featureWeightVector.entrySet()) {
      for (Map.Entry<String, FloatVector> feature : familyMap.getValue().entrySet()) {
        ModelRecord record = new ModelRecord();
        record.setFeatureFamily(familyMap.getKey());
        record.setFeatureName(feature.getKey());
        ArrayList<Double> arrayList = new ArrayList<>();
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
    labelWeightVector = new HashMap<>();

    embeddingDimension = header.getLabelEmbedding().entrySet().iterator().next().getValue().size();

    for (Map.Entry<String, java.util.List<Double>> labelRepresentation : header.getLabelEmbedding().entrySet()) {
      java.util.List<Double> values = labelRepresentation.getValue();
      String labelKey = labelRepresentation.getKey();
      FloatVector labelWeight = new FloatVector(embeddingDimension);
      for (int i = 0; i < embeddingDimension; i++) {
        labelWeight.set(i, values.get(i).floatValue());
      }
      labelWeightVector.put(labelKey, labelWeight);
    }

    featureWeightVector = new HashMap<>();
    for (long i = 0; i < rows; i++) {
      String line = reader.readLine();
      ModelRecord record = Util.decodeModel(line);
      String family = record.getFeatureFamily();
      String name = record.getFeatureName();
      Map<String, FloatVector> inner = featureWeightVector.get(family);
      if (inner == null) {
        inner = new HashMap<>();
        featureWeightVector.put(family, inner);
      }
      FloatVector vec = new FloatVector(record.getWeightVector().size());
      for (int j = 0; j < record.getWeightVector().size(); j++) {
        vec.values[j] = record.getWeightVector().get(j).floatValue();
      }
      inner.put(name, vec);
    }
  }
}
