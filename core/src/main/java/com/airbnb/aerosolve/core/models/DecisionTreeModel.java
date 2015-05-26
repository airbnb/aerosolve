package com.airbnb.aerosolve.core.models;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Serializable;
import java.lang.StringBuilder;
import java.util.Map;
import java.util.List;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.PriorityQueue;
import java.util.AbstractMap;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.util.Util;
import com.airbnb.aerosolve.core.util.Spline;
import lombok.Getter;
import lombok.Setter;

// A simple decision tree model.
public class DecisionTreeModel extends AbstractModel {

  private static final long serialVersionUID = 3651061358422885379L;

  @Getter @Setter
  protected ArrayList<ModelRecord> stumps;

  public DecisionTreeModel() {
  }

  @Override
  public float scoreItem(FeatureVector combinedItem) {
    Map<String, Map<String, Double>> floatFeatures = Util.flattenFeature(combinedItem);
    if (stumps.isEmpty()) return 0.0f;

    ModelRecord stump = stumps.get(0);
    while (!stump.isSetFeatureWeight()) {
      boolean response = BoostedStumpsModel.getStumpResponse(stump, floatFeatures);
      if (response) {
        stump = stumps.get(stump.rightChild);
      } else {
        stump = stumps.get(stump.leftChild);
      }
    }
    return (float) stump.featureWeight;
  }

  @Override
  // Decision trees don't usually have debuggable components.
  public float debugScoreItem(FeatureVector combinedItem,
                              StringBuilder builder) {
    return 0.0f;
  }

  @Override
  // Decision trees don't usually have debuggable components.
  public List<DebugScoreRecord> debugScoreComponents(FeatureVector combinedItem) {
    List<DebugScoreRecord> scoreRecordsList = new ArrayList<>();
    return scoreRecordsList;
  }

  public void save(BufferedWriter writer) throws IOException {
    ModelHeader header = new ModelHeader();
    header.setModelType("decision_tree");
    long count = stumps.size();
    header.setNumRecords(count);
    ModelRecord headerRec = new ModelRecord();
    headerRec.setModelHeader(header);
    writer.write(Util.encode(headerRec));
    writer.newLine();
    for (ModelRecord rec : stumps) {
      writer.write(Util.encode(rec));
      writer.newLine();
    }
    writer.flush();
  }

  @Override
  protected void loadInternal(ModelHeader header, BufferedReader reader) throws IOException {
    long rows = header.getNumRecords();

    stumps = new ArrayList<>();
    for (long i = 0; i < rows; i++) {
      String line = reader.readLine();
      ModelRecord record = Util.decodeModel(line);
      stumps.add(record);
    }
  }
}
