package com.airbnb.aerosolve.core.models;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.*;
import java.util.Map.Entry;

import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.util.Util;
import com.google.common.hash.HashCode;
import lombok.Getter;
import lombok.Setter;
import org.apache.http.annotation.NotThreadSafe;

/**
 * A linear model backed by a hash map.
 */
@NotThreadSafe
public class LinearModel extends AbstractModel {

  @Getter @Setter
  protected Map<String, Map<String, Float>> weights;

  @Override
  public float scoreItem(FeatureVector combinedItem) {
    Map<String, Set<String>> stringFeatures = combinedItem.getStringFeatures();
    if (stringFeatures == null || weights == null) {
      return 0.0f;
    }
    float sum = 0.0f;
    for (Entry<String, Set<String>> entry : stringFeatures.entrySet()) {
      String family = entry.getKey();
      Map<String, Float> inner = weights.get(family);
      if (inner == null) {
        continue;
      }

      for (String value : entry.getValue()) {
        Float weight = inner.get(value);
        if (weight != null) {
          sum += weight;
        }
      }
    }
    return sum;
  }

  public static class EntryComparator implements Comparator<Entry<String, Float>> {
    @Override
    public int compare(Entry<String, Float> e1, Entry<String, Float> e2) {
      float v1 = Math.abs(e1.getValue());
      float v2 = Math.abs(e2.getValue());
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
  public float debugScoreItem(FeatureVector combinedItem,
                              StringBuilder builder) {
    Map<String, Set<String>> stringFeatures = combinedItem.getStringFeatures();
    if (stringFeatures == null || weights == null) {
      return 0.0f;
    }
    float sum = 0.0f;
    PriorityQueue<Entry<String, Float>> scores = new PriorityQueue<>(100, new EntryComparator());
    Map<String, Float> familyScores = new HashMap<>();
    for (Entry<String, Set<String>> entry : stringFeatures.entrySet()) {
      String family = entry.getKey();
      for (String value : entry.getValue()) {
        HashCode code = Util.getHashCode(family, value);
        Map<String, Float> inner = weights.get(family);
        if (inner != null) {
          Float weight = inner.get(value);
          if (weight != null) {
            String str = family + ':' + value + " = " + weight + '\n';
            if (familyScores.containsKey(family)) {
              Float wt = familyScores.get(family);
              familyScores.put(family, wt + weight);
            } else {
              familyScores.put(family, weight);
            }
            AbstractMap.SimpleEntry<String, Float> ent = new AbstractMap.SimpleEntry<String, Float>(
                str, weight);
            scores.add(ent);
            sum += weight;
          }
        }
      }
    }
    builder.append("Scores by family ===>\n");
    if (!familyScores.isEmpty()) {
      PriorityQueue<Entry<String, Float>> familyPQ = new PriorityQueue<>(10, new EntryComparator());
      for (Entry<String, Float> entry : familyScores.entrySet()) {
        familyPQ.add(entry);
      }
      while (!familyPQ.isEmpty()) {
        Entry<String, Float> entry = familyPQ.poll();
        builder.append(entry.getKey() + " = " + entry.getValue() + '\n');
      }
    }
    builder.append("Top 15 scores ===>\n");
    if (!scores.isEmpty()) {
      int count = 0;
      float subsum = 0.0f;
      while (!scores.isEmpty()) {
        Entry<String, Float> entry = scores.poll();
        builder.append(entry.getKey());
        float val = entry.getValue();
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
    // linear model takes only string features
    Map<String, Set<String>> stringFeatures = combinedItem.getStringFeatures();
    List<DebugScoreRecord> scoreRecordsList = new ArrayList<>();
    if (stringFeatures == null || weights == null) {
      return scoreRecordsList;
    }
    for (Entry<String, Set<String>> entry : stringFeatures.entrySet()) {
      String family = entry.getKey();
      Map<String, Float> inner = weights.get(family);
      if (inner == null) continue;
      for (String value : entry.getValue()) {
        DebugScoreRecord record = new DebugScoreRecord();
        Float weight = inner.get(value);
        if (weight != null) {
          record.setFeatureFamily(family);
          record.setFeatureName(value);
          // 1.0 if the string feature exists, 0.0 otherwise
          record.setFeatureValue(1.0);
          record.setFeatureWeight(weight);
          scoreRecordsList.add(record);
        }
      }
    }
    return scoreRecordsList;
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
    weights = new HashMap<>();
    for (long i = 0; i < rows; i++) {
      String line = reader.readLine();
      ModelRecord record = Util.decodeModel(line);
      String family = record.getFeatureFamily();
      String name = record.getFeatureName();
      Map<String, Float> inner = weights.get(family);
      if (inner == null) {
        inner = new HashMap<>();
        weights.put(family, inner);
      }
      float weight = (float) record.getFeatureWeight();
      inner.put(name, weight);
    }
  }

  // save model
  public void save(BufferedWriter writer) throws IOException {
    ModelHeader header = new ModelHeader();
    header.setModelType("linear");
    header.setSlope(slope);
    header.setOffset(offset);
    long count = 0;
    for (Map.Entry<String, Map<String, Float>> familyMap : weights.entrySet()) {
      for (Map.Entry<String, Float> feature : familyMap.getValue().entrySet()) {
        count++;
      }
    }
    header.setNumRecords(count);
    ModelRecord headerRec = new ModelRecord();
    headerRec.setModelHeader(header);
    writer.write(Util.encode(headerRec));
    writer.newLine();
    for (Map.Entry<String, Map<String, Float>> familyMap : weights.entrySet()) {
      for (Map.Entry<String, Float> feature : familyMap.getValue().entrySet()) {
        ModelRecord record = new ModelRecord();
        record.setFeatureFamily(familyMap.getKey());
        record.setFeatureName(feature.getKey());
        record.setFeatureWeight(feature.getValue());
        writer.write(Util.encode(record));
        writer.newLine();
      }
    }
    writer.flush();
  }
}
