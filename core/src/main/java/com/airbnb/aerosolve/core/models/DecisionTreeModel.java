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
    return scoreFlattenedFeature(floatFeatures);
  }
  
  public float scoreFlattenedFeature(Map<String, Map<String, Double>> floatFeatures) {
    int leaf = getLeafIndex(floatFeatures);
    if (leaf < 0) return 0.0f;

    ModelRecord stump = stumps.get(leaf);
    return (float) stump.featureWeight;
  }

  public int getLeafIndex(Map<String, Map<String, Double>> floatFeatures) {
    if (stumps.isEmpty()) return -1;

    int index = 0;
    while (true) {
      ModelRecord stump = stumps.get(index);
      if (!stump.isSetLeftChild() || !stump.isSetRightChild()) {
        break;
      }
      boolean response = BoostedStumpsModel.getStumpResponse(stump, floatFeatures);
      if (response) {
        index = stump.rightChild;
      } else {
        index = stump.leftChild;
      }
    }
    return index;
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

  /*
   * Returns a debuggable single tree in graphviz DOT format
   */
  public String toDot() {
    StringBuilder sb = new StringBuilder();
    sb.append("digraph g {\n");
    sb.append("graph [ rankdir = \"LR\" ]\n");
    for (int i = 0; i < stumps.size(); i++) {
      ModelRecord stump = stumps.get(i);
      if (stump.isSetLeftChild()) {
        sb.append(String.format("\"node%d\" [\n", i));
        double thresh = stump.threshold;
        sb.append(String.format(
            "label = \"<f0> %s:%s | <f1> less or equal %f | <f2> greater than %f\";\n",
            stump.featureFamily,
            stump.featureName,
            thresh,
            thresh));
        sb.append("shape = \"record\";\n");
        sb.append("];\n");
      } else {
        sb.append(String.format("\"node%d\" [\n", i));
        sb.append(String.format("label = \"<f0> Weight %f\";\n", stump.featureWeight));
        sb.append("shape = \"record\";\n");
        sb.append("];\n");
      }
    }
    int count = 0;
    for (int i = 0; i < stumps.size(); i++) {
      ModelRecord stump = stumps.get(i);
      if (stump.isSetLeftChild()) {
        sb.append(String.format("\"node%d\":f1 -> \"node%d\":f0 [ id = %d ];\n", i, stump.leftChild, count));
        count = count  + 1;
        sb.append(String.format("\"node%d\":f2 -> \"node%d\":f0 [id = %d];\n", i, stump.rightChild, count));
        count = count + 1;
      }
    }
    sb.append("}\n");
    return sb.toString();
  }

  // Returns the transform config in human readable form.
  public String toHumanReadableTransform() {
    StringBuilder sb = new StringBuilder();
    sb.append("  nodes: [\n");
    for (int i = 0; i < stumps.size(); i++) {
      ModelRecord stump = stumps.get(i);
      sb.append("    \"");
      if (stump.isSetLeftChild()) {
        // Parent node, node id, family, name, threshold, left, right  
        sb.append(
            String.format("P,%d,%s,%s,%f,%d,%d", i,
                stump.featureFamily,
                stump.featureName,
                stump.threshold,
                stump.leftChild, stump.rightChild));
      } else {
        // Leaf node, node id, feature weight, human readable leaf name.  
        sb.append(String.format("L,%d,%f,LEAF_%d", i, stump.featureWeight, i));  
      }
      sb.append("\"\n");
    }
    sb.append("  ]\n");
    return sb.toString();
  }

  // Constructs a tree from human readable transform list.
  public static DecisionTreeModel fromHumanReadableTransform(List<String> rows) {
    DecisionTreeModel tree = new DecisionTreeModel();  
    ArrayList<ModelRecord> records = new ArrayList<>();
    tree.setStumps(records);
    for (String row : rows) {
      ModelRecord rec = new ModelRecord();
      records.add(rec);
      String token[] = row.split(",");
      if (token[0].contains("P")) {
        // Parent node
        rec.setFeatureFamily(token[2]);
        rec.setFeatureName(token[3]);
        rec.setThreshold(Double.parseDouble(token[4]));
        rec.setLeftChild(Integer.parseInt(token[5]));
        rec.setRightChild(Integer.parseInt(token[6]));
      } else {
        rec.setFeatureName(token[3]);
        rec.setFeatureWeight(Double.parseDouble(token[2]));
      }
    }
    return tree;
  }
}
