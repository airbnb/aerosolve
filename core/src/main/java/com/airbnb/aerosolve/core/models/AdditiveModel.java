package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.function.AbstractFunction;
import com.airbnb.aerosolve.core.function.Function;
import com.airbnb.aerosolve.core.function.Linear;
import com.airbnb.aerosolve.core.function.Spline;
import com.airbnb.aerosolve.core.perf.Feature;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.perf.FeatureValue;
import com.airbnb.aerosolve.core.util.Util;
import it.unimi.dsi.fastutil.objects.Reference2ObjectMap;
import it.unimi.dsi.fastutil.objects.Reference2ObjectOpenHashMap;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

// A generalized additive model with a parametric function per feature.
// See http://en.wikipedia.org/wiki/Generalized_additive_model
@Slf4j
public class AdditiveModel extends AbstractModel {
  private static final String DENSE_FAMILY = "dense";

  @Getter @Setter
  private Reference2ObjectMap<Feature, Function> weights =
      new Reference2ObjectOpenHashMap<>();

  // only MultiDimensionSpline using denseWeights
  // whole dense features belongs to feature family DENSE_FAMILY
  private Map<String, Function> denseWeights;

  public AdditiveModel(FeatureRegistry registry) {
    super(registry);
  }

  // TODO (Brad): Fix
  /*private Map<String, Function> getOrCreateDenseWeights() {
    if (denseWeights == null) {
      denseWeights = weights.get(DENSE_FAMILY);
      if (denseWeights == null) {
        denseWeights = new HashMap<>();
        weights.put(DENSE_FAMILY, denseWeights);
      }
    }
    return denseWeights;
  }*/

  @Override
  public boolean needsFeature(Feature feature) {
    return weights.containsKey(feature);
  }

  @Override
  public double scoreItem(FeatureVector combinedItem) {
    double sum = 0.0d;

    for (FeatureValue value : combinedItem) {
      Function func = weights.get(value.feature());
      if (func == null)
        continue;
      sum += func.evaluate(value.getDoubleValue());
    }
    return sum;
  }

  @Override
  public double debugScoreItem(FeatureVector combinedItem,
                              StringBuilder builder) {
    double sum = 0.0d;
    // order by the absolute value
    PriorityQueue<Map.Entry<FeatureValue, Double>> scores =
        new PriorityQueue<>(100, new LinearModel.EntryComparator());

    for (FeatureValue value : combinedItem) {
      Function func = weights.get(value.feature());
      if (func == null)
        continue;
      double subscore = func.evaluate(value.getDoubleValue());
      sum += subscore;
      scores.add(new AbstractMap.SimpleEntry<FeatureValue, Double>(value, subscore));
    }

    final int MAX_COUNT = 100;
    builder.append("Top scores ===>\n");
    if (!scores.isEmpty()) {
      int count = 0;
      double subsum = 0.0d;
      while (!scores.isEmpty()) {
        Map.Entry<FeatureValue, Double> entry = scores.poll();
        FeatureValue value = entry.getKey();
        String str = value.feature().family().name() + ":" + value.feature().name() +
                     "=" + value.getDoubleValue() + " = " + entry.getValue() + "<br>\n";
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

    for (FeatureValue value : combinedItem) {
      Function func = weights.get(value.feature());
      if (func == null) continue;
      double weight = func.evaluate(value.getDoubleValue());
      DebugScoreRecord record = new DebugScoreRecord();
      record.setFeatureFamily(value.feature().family().name());
      record.setFeatureName(value.feature().name());
      record.setFeatureValue(value.getDoubleValue());
      record.setFeatureWeight(weight);
      scoreRecordsList.add(record);
    }

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
      Feature feature = registry.feature(family, name);
      FunctionForm funcForm = record.getFunctionForm();
      if (funcForm == FunctionForm.SPLINE) {
        weights.put(feature, new Spline(record));
      } else if (funcForm == FunctionForm.LINEAR) {
        weights.put(feature, new Linear(record));
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
    writer.flush();
  }

  public void addFunction(String featureFamily, String featureName, FunctionForm functionForm,
                          double[] params, boolean overwrite) {
    // For SPLINE: params[0] = minValue, params[1] = maxValue, params[2] = numBin
    // For LINEAR: params[0] = minValue, params[1] = maxValue
    // overwrite: if TRUE, overwrite existing feature function
    Feature feature = registry.feature(featureFamily, featureName);
    double minVal = params[0];
    double maxVal = params[1];
    if (functionForm == FunctionForm.SPLINE) {
      int numBins = (int) params[2];
      if (maxVal <= minVal) {
        maxVal = minVal + 1.0d;
      }
      Spline spline = new Spline(minVal, maxVal, new double[numBins]);
      if (overwrite) {
        weights.put(feature, spline);
      } else {
        weights.putIfAbsent(feature, spline);
      }
    } else if (functionForm == FunctionForm.LINEAR) {
      Linear linear = new Linear(minVal, maxVal, new double[2]);
      if (overwrite) {
        weights.put(feature, linear);
      } else {
        weights.putIfAbsent(feature, linear);
      }
    }
  }

  // Update weights based on gradient and learning rate
  public void update(double grad,
                     double learningRate,
                     double cap,
                     FeatureVector vector) {
    // update with lInfinite cap
    for (FeatureValue value : vector) {
      Function func = weights.get(value.feature());
      if (func == null) continue;
      func.update(value.getDoubleValue(), -grad * learningRate);
      func.LInfinityCap(cap);
    }
  }
}
