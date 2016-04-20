package com.airbnb.aerosolve.core.models;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.util.Util;
import lombok.Getter;
import lombok.Setter;

// A simple boosted decision stump model that only operates on float features.
public class BoostedStumpsModel extends AbstractModel {

  private static final long serialVersionUID = 3651061358422885377L;

  @Getter @Setter
  protected List<ModelRecord> stumps;

  public BoostedStumpsModel() {
  }

  // Returns true if >= stump, false otherwise.
  public static boolean getStumpResponse(ModelRecord stump,
                                         Map<String, Map<String, Double>> floatFeatures) {
    Map<String, Double> feat = floatFeatures.get(stump.featureFamily);
    // missing feature corresponding to false (left branch)
    if (feat == null) {
      return false;
    }
    Double val = feat.get(stump.featureName);
    if (val == null) {
      return false;
    }
    if (val >= stump.getThreshold()) {
      return true;
    } else {
      return false;
    }
  }

  @Override
  public float scoreItem(FeatureVector combinedItem) {
    float sum = 0.0f;
    Map<String, Map<String, Double>> floatFeatures = Util.flattenFeature(combinedItem);
    for (ModelRecord stump : stumps) {
      if (getStumpResponse(stump, floatFeatures)) {
        sum += stump.featureWeight;
      }
    }
    return sum;
  }

  @Override
  public float debugScoreItem(FeatureVector combinedItem,
                              StringBuilder builder) {
    float sum = 0.0f;
    Map<String, Map<String, Double>> floatFeatures = Util.flattenFeature(combinedItem);
    for (ModelRecord stump : stumps) {
      boolean response = getStumpResponse(stump, floatFeatures);
      String output = stump.featureFamily + ':' + stump.getFeatureName();
      Double threshold = stump.threshold;
      Double weight = stump.featureWeight;
      if (response) {
        builder.append(output);
        builder.append(" >= " + threshold.toString() + " ==> " + weight.toString());
        sum += stump.featureWeight;
      }
    }
    return sum;
  }

  @Override
  public List<DebugScoreRecord> debugScoreComponents(FeatureVector combinedItem) {
    List<DebugScoreRecord> scoreRecordsList = new ArrayList<>();
    Map<String, Map<String, Double>> floatFeatures = Util.flattenFeature(combinedItem);

    for (ModelRecord stump : stumps) {
      boolean response = getStumpResponse(stump, floatFeatures);

      if (response) {
        DebugScoreRecord record = new DebugScoreRecord();
        record.setFeatureFamily(stump.featureFamily);
        record.setFeatureName(stump.featureName);
        record.setFeatureValue(floatFeatures.get(stump.featureFamily).get(stump.featureName));
        record.setFeatureWeight(stump.featureWeight);
        scoreRecordsList.add(record);
      }
    }

    return scoreRecordsList;
  }

  public void save(BufferedWriter writer) throws IOException {
    ModelHeader header = new ModelHeader();
    header.setModelType("boosted_stumps");
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
