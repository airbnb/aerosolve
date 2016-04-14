package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.FamilyVector;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.functions.AbstractFunction;
import com.airbnb.aerosolve.core.functions.Function;
import com.airbnb.aerosolve.core.functions.MultiDimensionSpline;
import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.util.Util;
import com.google.common.primitives.Doubles;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.objects.Reference2ObjectOpenHashMap;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.tuple.Pair;

// A generalized additive model with a parametric function per feature.
// See http://en.wikipedia.org/wiki/Generalized_additive_model
@Slf4j
@Accessors(fluent = true, chain = true)
public class AdditiveModel extends AbstractModel {

  @Getter @Setter
  private Map<Feature, Function> weights =
      new Object2ObjectOpenHashMap<>();

  @Getter
  private Map<Family, Function> familyWeights =
      new Object2ObjectOpenHashMap<>();

  public AdditiveModel(FeatureRegistry registry) {
    super(registry);
  }

  @Override
  public boolean needsFeature(Feature feature) {
    return weights.containsKey(feature);
  }

  @Override
  public double scoreItem(FeatureVector combinedItem) {
    return scoreItemInternal(combinedItem, null, null);
  }

  public double scoreItemInternal(FeatureVector combinedItem,
                                  PriorityQueue<Map.Entry<String, Double>> scores,
                                  List<DebugScoreRecord> scoreRecordsList) {
    double sum = 0.0d;

    if (combinedItem instanceof MultiFamilyVector && familyWeights != null
        && !familyWeights.isEmpty()) {
      MultiFamilyVector multiFamilyVector = ((MultiFamilyVector) combinedItem);
      for (FamilyVector familyVector : multiFamilyVector.families()) {
        Function familyFunction = familyWeights.get(familyVector.family());

        if (familyFunction == null) {
          sum += scoreVector(familyVector, scores, scoreRecordsList);
        } else {
          double[] val = familyVector.denseArray();
          double subscore = familyFunction.evaluate(val);
          sum += subscore;
          if (scores != null) {
            String str = familyVector.family().name() + ":null=" + Arrays.toString(val)
                         + " = " + subscore + "<br>\n";
            scores.add(Pair.of(str, subscore));
          }
          if (scoreRecordsList != null) {
            DebugScoreRecord record = new DebugScoreRecord();
            record.setFeatureFamily(familyVector.family().name());
            record.setFeatureName(null);
            record.setDenseFeatureValue(Doubles.asList(val));
            record.setFeatureWeight(subscore);
            scoreRecordsList.add(record);
          }
        }
      }
    } else {
      sum = scoreVector(combinedItem, scores, scoreRecordsList);
    }

    return sum;
  }

  private double scoreVector(FeatureVector combinedItem,
                             PriorityQueue<Map.Entry<String, Double>> scores,
                             List<DebugScoreRecord> scoreRecordsList) {
    double sum = 0.0d;
    for (FeatureValue value : combinedItem) {
      Function func = weights.get(value.feature());
      if (func == null)
        continue;
      double subscore = func.evaluate(value.value());
      sum += subscore;
      if (scores != null) {
        String str = value.feature().family().name() + ":" + value.feature().name() +
                     "=" + value.value() + " = " +subscore + "<br>\n";
        scores.add(Pair.of(str, subscore));
      }
      if (scoreRecordsList != null) {
        DebugScoreRecord record = new DebugScoreRecord();
        record.setFeatureFamily(value.feature().family().name());
        record.setFeatureName(value.feature().name());
        record.setFeatureValue(value.value());
        record.setFeatureWeight(subscore);
        scoreRecordsList.add(record);
      }
    }
    return sum;
  }

  @Override
  public double debugScoreItem(FeatureVector combinedItem,
                              StringBuilder builder) {
    // order by the absolute value
    PriorityQueue<Map.Entry<String, Double>> scores =
        new PriorityQueue<>(100, new LinearModel.EntryComparator());

    double sum = scoreItemInternal(combinedItem, scores, null);

    final int MAX_COUNT = 100;
    builder.append("Top scores ===>\n");
    if (!scores.isEmpty()) {
      int count = 0;
      double subsum = 0.0d;
      while (!scores.isEmpty()) {
        Map.Entry<String, Double> entry = scores.poll();
        String str = entry.getKey();
        builder.append(str);
        double val = entry.getValue();
        subsum += val;
        count = count + 1;
        if (count >= MAX_COUNT) {
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
    scoreItemInternal(combinedItem, null, scoreRecordsList);
    return scoreRecordsList;
  }

  @Override
  protected void loadInternal(ModelHeader header, BufferedReader reader) throws IOException {
    long rows = header.getNumRecords();
    slope = header.getSlope();
    offset = header.getOffset();
    weights = new Reference2ObjectOpenHashMap<>();
    for (long i = 0; i < rows; i++) {
      String line = reader.readLine();
      ModelRecord record = Util.decodeModel(line);
      String family = record.getFeatureFamily();
      String name = record.getFeatureName();
      Function function = AbstractFunction.buildFunction(record);
      // TODO (Brad): Do we need to check the type? I did it for safety in case some null names have
      // been serialized. But it's a bit brittle to use instanceof.
      if (name == null && function instanceof MultiDimensionSpline) {
        familyWeights.put(registry.family(family), function);
      } else {
        weights.put(registry.feature(family, name), function);
      }
    }
  }

  @Override
  public void save(BufferedWriter writer) throws IOException {
    ModelHeader header = new ModelHeader();
    header.setModelType("additive");
    header.setSlope(slope);
    header.setOffset(offset);
    header.setNumRecords(weights.size());
    ModelRecord headerRec = new ModelRecord();
    headerRec.setModelHeader(header);
    writer.write(Util.encode(headerRec));
    writer.newLine();
    for (Map.Entry<Feature, Function> entry : weights.entrySet()) {
      Function func = entry.getValue();
      String featureName = entry.getKey().name();
      writer.write(Util.encode(func.toModelRecord(entry.getKey().family().name(), featureName)));
      writer.newLine();
    }
    // TODO (Brad): Talk to Julian.  This changes the serialization from what he was using but
    // I don't think it should be a problem since we're still training.
    for (Map.Entry<Family, Function> entry : familyWeights.entrySet()) {
      Function func = entry.getValue();
      writer.write(Util.encode(func.toModelRecord(entry.getKey().name(), null)));
      writer.newLine();
    }
    writer.flush();
  }

  public void addFunction(Feature feature, Function function, boolean overwrite) {
    if (function == null) {
      throw new RuntimeException(feature + " function null");
    }
    if (overwrite || !weights.containsKey(feature)) {
      weights.put(feature, function);
    }
  }

  public void addFunction(Family family, Function function, boolean overwrite) {
    if (function == null) {
      throw new RuntimeException(family + " function null");
    }
    if (overwrite || !familyWeights.containsKey(family)) {
      familyWeights.put(family, function);
    }
  }

  // Update weights based on gradient and learning rate
  public void update(double gradWithLearningRate,
                     double cap,
                     MultiFamilyVector vector) {
    for (FamilyVector familyVector : vector.families()) {
      Function func = familyWeights.get(familyVector.family());
      if (func == null) {
        for (FeatureValue value : familyVector) {
          func = weights.get(value.feature());
          if (func == null) continue;
          // update with lInfinite cap
          func.update(-gradWithLearningRate, value.value());
          func.LInfinityCap(cap);
        }
      } else {
        // update with lInfinite cap
        func.update(-gradWithLearningRate, vector.denseArray());
        func.LInfinityCap(cap);
      }
    }
  }
}
