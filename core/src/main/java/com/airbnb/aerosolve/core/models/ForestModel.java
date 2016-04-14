package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.MulticlassScoringResult;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.util.Util;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;

// A tree forest model.
@Accessors(fluent = true, chain = true)
public class ForestModel extends AbstractModel {

  private static final long serialVersionUID = 3651061358422885378L;

  @Getter @Setter
  protected ArrayList<DecisionTreeModel> trees;

  public ForestModel(FeatureRegistry registry) {
    super(registry);
  }

  @Override
  public double scoreItem(FeatureVector combinedItem) {
    float sum = 0.0f;
    // Note: we sum instead of average so that the trainer has the option of boosting the
    // trees together.
    for (DecisionTreeModel tree : trees) {
      sum += tree.scoreItem(combinedItem);
    }
    return sum;
  }

  @Override
  public ArrayList<MulticlassScoringResult> scoreItemMulticlass(FeatureVector combinedItem) {
    HashMap<String, Double> map = new HashMap<>();

    // Note: we sum instead of average so that the trainer has the option of boosting the
    // trees together.
    for (DecisionTreeModel tree : trees) {
      ArrayList<MulticlassScoringResult> tmp = tree.scoreItemMulticlass(combinedItem);
      for (MulticlassScoringResult result : tmp) {
        Double v = map.get(result.getLabel());
        if (v == null) {
          map.put(result.getLabel(), result.getScore());
        } else {
          map.put(result.getLabel(), v + result.getScore());
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
  public double debugScoreItem(FeatureVector combinedItem,
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
      DecisionTreeModel tree = new DecisionTreeModel(registry);
      tree.loadInternal(record.getModelHeader(), reader);
      trees.add(tree);
    }
  }
}
