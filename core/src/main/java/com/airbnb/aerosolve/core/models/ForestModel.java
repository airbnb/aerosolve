package com.airbnb.aerosolve.core.models;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.lang.StringBuilder;
import java.util.Map;
import java.util.List;
import java.util.HashMap;
import java.util.ArrayList;

import com.airbnb.aerosolve.core.*;
import com.airbnb.aerosolve.core.util.Util;
import lombok.Getter;
import lombok.Setter;

// A tree forest model.
public class ForestModel extends AbstractModel {

  private static final long serialVersionUID = 3651061358422885378L;

  @Getter @Setter
  protected ArrayList<DecisionTreeModel> trees;

  public ForestModel() {
  }

  @Override
  public float scoreItem(FeatureVector combinedItem) {
    Map<String, Map<String, Double>> floatFeatures = Util.flattenFeature(combinedItem);

    float sum = 0.0f;
    // Note: we sum instead of average so that the trainer has the option of boosting the
    // trees together.
    for (int i = 0; i < trees.size(); i++) {
      sum += trees.get(i).scoreFlattenedFeature(floatFeatures);
    }
    return sum;
  }

  @Override
  public ArrayList<MulticlassScoringResult> scoreItemMulticlass(FeatureVector combinedItem) {
    HashMap<String, Double> map = new HashMap<>();

    Map<String, Map<String, Double>> floatFeatures = Util.flattenFeature(combinedItem);

    // Note: we sum instead of average so that the trainer has the option of boosting the
    // trees together.
    for (int i = 0; i < trees.size(); i++) {
      ArrayList<MulticlassScoringResult> tmp = trees.get(i).scoreFlattenedFeatureMulticlass(
          floatFeatures);
      for (MulticlassScoringResult result : tmp) {
        Double v = map.get(result.label);
        if (v == null) {
          map.put(result.label, result.score);
        } else {
          map.put(result.label, v + result.score);
        }
      }
    }

    ArrayList<MulticlassScoringResult> results =  new ArrayList<>();
    for (Map.Entry<String, Double> entry : map.entrySet()) {
      MulticlassScoringResult result = new MulticlassScoringResult();
      result.setLabel(entry.getKey());
      result.setScore(entry.getValue());
      results.add(result);
    }

    return results;
  }

  @Override
  // Forests don't usually have debuggable components.
  public float debugScoreItem(FeatureVector combinedItem,
      StringBuilder builder) {
    return 0.0f;
  }

  @Override
  // Forests don't usually have debuggable components.
  public List<DebugScoreRecord> debugScoreComponents(FeatureVector combinedItem) {
    List<DebugScoreRecord> scoreRecordsList = new ArrayList<>();
    return scoreRecordsList;
  }

  public void save(BufferedWriter writer) throws IOException {
    ModelHeader header = new ModelHeader();
    header.setModelType("forest");
    long count = trees.size();
    header.setNumRecords(count);
    header.setSlope(slope);
    header.setOffset(offset);
    ModelRecord headerRec = new ModelRecord();
    headerRec.setModelHeader(header);
    writer.write(Util.encode(headerRec));
    writer.newLine();
    for (DecisionTreeModel tree : trees) {
      tree.save(writer);
    }
    writer.flush();
  }

  @Override
  protected void loadInternal(ModelHeader header, BufferedReader reader) throws IOException {
    long numTrees = header.getNumRecords();
    slope = header.getSlope();
    offset = header.getOffset();
    trees = new ArrayList<>();
    for (long i = 0; i < numTrees; i++) {
      String line = reader.readLine();
      ModelRecord record = Util.decodeModel(line);
      DecisionTreeModel tree = new DecisionTreeModel();
      tree.loadInternal(record.getModelHeader(), reader);
      trees.add(tree);
    }
  }
}
