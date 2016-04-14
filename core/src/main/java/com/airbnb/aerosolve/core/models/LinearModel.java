package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.util.Util;
import it.unimi.dsi.fastutil.objects.Object2DoubleMap;
import it.unimi.dsi.fastutil.objects.Object2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.objects.Reference2DoubleMap;
import it.unimi.dsi.fastutil.objects.Reference2DoubleOpenHashMap;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import lombok.Getter;
import lombok.experimental.Accessors;
import org.apache.http.annotation.NotThreadSafe;

/**
 * A linear model backed by a hash map.
 */
@NotThreadSafe
@Accessors(fluent = true, chain = true)
public class LinearModel extends AbstractModel {

  @Getter
  protected Object2DoubleMap<Feature> weights;

  public LinearModel(FeatureRegistry registry) {
    super(registry);
    weights = new Object2DoubleOpenHashMap<>();
  }

  @Override
  public double scoreItem(FeatureVector combinedItem) {
    return scoreItemInternal(combinedItem, null, null, null);
  }

  public static class EntryComparator implements Comparator<Entry<?, Double>> {
    @Override
    public int compare(Entry<?, Double> e1, Entry<?, Double> e2) {
      double v1 = Math.abs(e1.getValue());
      double v2 = Math.abs(e2.getValue());
      if (v1 > v2) {
        return -1;
      } else if (v1 < v2) {
        return 1;
      }
      return 0;
    }
  }

  // Debug scores a single item. These are explanations for why a model
  // came up with the score.
  @Override
  public double debugScoreItem(FeatureVector combinedItem,
                              StringBuilder builder) {
    if (weights == null) {
      return 0.0d;
    }

    PriorityQueue<Entry<Feature, Double>> scores = new PriorityQueue<>(100, new EntryComparator());
    Reference2DoubleMap<Family> familyScores = new Reference2DoubleOpenHashMap<>();
    double sum = scoreItemInternal(combinedItem, familyScores, scores, null);

    builder.append("Scores by family ===>\n");
    if (!familyScores.isEmpty()) {
      PriorityQueue<Entry<Family, Double>> familyPQ = new PriorityQueue<>(10, new EntryComparator());
      for (Entry<Family, Double> entry : familyScores.entrySet()) {
        familyPQ.add(entry);
      }
      while (!familyPQ.isEmpty()) {
        Entry<Family, Double> entry = familyPQ.poll();
        builder.append(entry.getKey().name() + " = " + entry.getValue() + '\n');
      }
    }
    builder.append("Top 15 scores ===>\n");
    if (!scores.isEmpty()) {
      int count = 0;
      float subsum = 0.0f;
      while (!scores.isEmpty()) {
        Entry<Feature, Double> entry = scores.poll();
        Feature feature = entry.getKey();
        double val = entry.getValue();
        String str = feature.family().name() + ':' + feature.name() +
                     " = " + val + '\n';
        builder.append(str);
        subsum += val;
        count = count + 1;
        if (count >= 15) {
          builder.append("Leftover = " + (sum - subsum) + '\n');
          break;
        }
      }
    }
    builder.append("Total = " + sum + '\n');
    return sum;
  }

  @Override
  public List<DebugScoreRecord> debugScoreComponents(FeatureVector combinedItem) {
    List<DebugScoreRecord> scoreRecordsList = new ArrayList<>();
    scoreItemInternal(combinedItem, null, null, scoreRecordsList);
    return scoreRecordsList;
  }

  private double scoreItemInternal(FeatureVector combinedItem,
                                   Reference2DoubleMap<Family> familyScores,
                                   PriorityQueue<Entry<Feature, Double>> scores,
                                   List<DebugScoreRecord> scoreRecordsList) {
    if (weights == null) {
      return 0.0d;
    }
    double sum = 0.0d;

    // No need to filter to string values.  Just iterate the keys and see if they exist in the model
    // If we passed in a vector with values that matter then we made a mistake using a linear model.
    for (Feature feature : combinedItem.keySet()) {
      if (!weights.containsKey(feature)) {
        continue;
      }
      double weight = weights.getDouble(feature);
      sum += weight;

      if (familyScores != null) {
        Family family = feature.family();
        if (familyScores.containsKey(family)) {
          double wt = familyScores.getDouble(family);
          familyScores.put(family, wt + weight);
        } else {
          familyScores.put(family, weight);
        }
      }

      if (scores != null) {
        AbstractMap.SimpleEntry<Feature, Double> ent = new AbstractMap.SimpleEntry<>(
            feature, weight);
        scores.add(ent);
      }

      if (scoreRecordsList != null) {
        DebugScoreRecord record = new DebugScoreRecord();
        record.setFeatureFamily(feature.family().name());
        record.setFeatureName(feature.name());
        // 1.0 if the string feature exists, 0.0 otherwise
        record.setFeatureValue(1.0);
        record.setFeatureWeight(weight);
        scoreRecordsList.add(record);
      }
    }
    return sum;
  }

  // Loads model from a buffered stream.
  @Override
  protected void loadInternal(ModelHeader header, BufferedReader reader) throws IOException {
    long rows = header.getNumRecords();
    // Very old models did not set slope and offset so check first.
    if (header.isSetSlope()) {
      slope = header.getSlope();      
    }
    if (header.isSetOffset()) {
      offset = header.getOffset();      
    }
    weights = new Object2DoubleOpenHashMap<>();
    for (long i = 0; i < rows; i++) {
      String line = reader.readLine();
      ModelRecord record = Util.decodeModel(line);
      String family = record.getFeatureFamily();
      String name = record.getFeatureName();
      Feature feature = registry.feature(family, name);
      double weight = record.getFeatureWeight();
      weights.put(feature, weight);
    }
  }

  // save model
  public void save(BufferedWriter writer) throws IOException {
    ModelHeader header = new ModelHeader();
    header.setModelType("linear");
    header.setSlope(slope);
    header.setOffset(offset);
    header.setNumRecords(weights.size());
    ModelRecord headerRec = new ModelRecord();
    headerRec.setModelHeader(header);
    writer.write(Util.encode(headerRec));
    writer.newLine();
    for (Object2DoubleMap.Entry<Feature> entry : weights.object2DoubleEntrySet()) {
      Feature feature = entry.getKey();
      ModelRecord record = new ModelRecord();
      record.setFeatureFamily(feature.family().name());
      record.setFeatureName(feature.name());
      record.setFeatureWeight(entry.getDoubleValue());
      writer.write(Util.encode(record));
      writer.newLine();
    }
    writer.flush();
  }
}
