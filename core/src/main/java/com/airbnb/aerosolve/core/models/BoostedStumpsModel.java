package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.util.Util;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;

// A simple boosted decision stump model that only operates on float features.
@Accessors(fluent = true, chain = true)
public class BoostedStumpsModel extends AbstractModel {

  private static final long serialVersionUID = 3651061358422885377L;

  @Getter @Setter
  protected List<ModelRecord> stumps;

  public BoostedStumpsModel(FeatureRegistry registry) {
    super(registry);
  }

  // Returns true if >= stump, false otherwise.
  public static boolean getStumpResponse(ModelRecord stump, FeatureVector vector) {
    Feature feature = vector.registry().feature(stump.getFeatureFamily(), stump.getFeatureName());
    return vector.containsKey(feature) &&
           vector.getDouble(feature) >= stump.getThreshold();
  }

  @Override
  public double scoreItem(FeatureVector combinedItem) {
    float sum = 0.0f;
    for (ModelRecord stump : stumps) {
      if (getStumpResponse(stump, combinedItem)) {
        sum += stump.getFeatureWeight();
      }
    }
    return sum;
  }

  @Override
  public double debugScoreItem(FeatureVector combinedItem,
                              StringBuilder builder) {
    float sum = 0.0f;
    for (ModelRecord stump : stumps) {
      boolean response = getStumpResponse(stump, combinedItem);
      String output = stump.getFeatureFamily() + ':' + stump.getFeatureName();
      Double threshold = stump.getThreshold();
      Double weight = stump.getFeatureWeight();
      if (response) {
        builder.append(output);
        builder.append(" >= " + threshold.toString() + " ==> " + weight.toString());
        sum += stump.getFeatureWeight();
      }
    }
    return sum;
  }

  @Override
  public List<DebugScoreRecord> debugScoreComponents(FeatureVector combinedItem) {
    List<DebugScoreRecord> scoreRecordsList = new ArrayList<>();

    for (ModelRecord stump : stumps) {
      boolean response = getStumpResponse(stump, combinedItem);

      if (response) {
        Feature feature = registry.feature(stump.getFeatureFamily(), stump.getFeatureName());
        DebugScoreRecord record = new DebugScoreRecord();
        record.setFeatureFamily(stump.getFeatureFamily());
        record.setFeatureName(stump.getFeatureName());
        record.setFeatureValue(combinedItem.get(feature));
        record.setFeatureWeight(stump.getFeatureWeight());
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
