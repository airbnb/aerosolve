package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.features.SparseLabeledPoint;
import com.airbnb.aerosolve.core.function.AbstractFunction;
import com.airbnb.aerosolve.core.function.Function;
import com.airbnb.aerosolve.core.util.Util;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;

import static com.airbnb.aerosolve.core.function.FunctionUtil.toFloat;

/**
 * A generalized additive model with a parametric function per feature. See
 * http://en.wikipedia.org/wiki/Generalized_additive_model
 *
 * Aside from common functionality AdditiveModel has special optimization that uses an indexer for
 * features. This allows efficient storage of sparse feature vector such that they could be persist
 * into memory or disk if desired. This is achieved by calling `generateFeatureIndexer` function
 * after model has been initialized with all feature weights, which arranges all known features into
 * an array. All feature vector can now be flatten into a sparse vector according to index of
 * feature in corresponding feature indexer. Subsequent model update can then be made via array
 * lookup instead of nested map lookup. This is done via calling `generateWeightVector` to populate
 * the array of function corresponding to the feature.
 */
@Slf4j
public class AdditiveModel extends AbstractModel implements Cloneable {
  public static final String DENSE_FAMILY = "dense";
  @Getter
  @Setter
  private Map<String, Map<String, Function>> weights = new HashMap<>();

  // only MultiDimensionSpline using denseWeights
  // whole dense features belongs to feature family DENSE_FAMILY
  private Map<String, Function> denseWeights;

  private Map<String, Function> getOrCreateDenseWeights() {
    if (denseWeights == null) {
      denseWeights = weights.get(DENSE_FAMILY);
      if (denseWeights == null) {
        denseWeights = weights.put(DENSE_FAMILY, new HashMap<>());
      }
    }
    return denseWeights;
  }

  // featureIndexer maps features to a unique consecutive ascending index
  // mapping training data via this index before shuffling can save a lot of time
  @Getter
  private Map<String, Map<String, Integer>> featureIndexer = new HashMap<>();
  // weightVector takes the index generated above and construct an indexed weight function vector
  @Getter
  private Function[] weightVector = new Function[0];

  /**
   * Generate the feature indexer which maps each feature to a unique integer index
   *
   * @apiNote Subsequent `generateWeightVector` calls will use this index. This index does not
   * automatically update when features are added or removed.
   */
  public AdditiveModel generateFeatureIndexer() {
    featureIndexer.clear();

    int count = 0;
    for (Map.Entry<String, Map<String, Function>> family : weights.entrySet()) {
      String familyName = family.getKey();
      Map<String, Integer> featureIndex = new HashMap<>();
      featureIndexer.put(familyName, featureIndex);
      for (Map.Entry<String, Function> feature : family.getValue().entrySet()) {
        featureIndex.put(feature.getKey(), count++);
      }
    }
    // (re)initialize the weight vector to be populated
    weightVector = new Function[count];
    return this;
  }

  /**
   * Populate the weight vector with index weight function according to feature indexer
   *
   * @apiNote `generateFeatureIndexer` must be called before this function if there is any feature
   * set modification No error/boundary checking is performed in this function. This function is
   * automatically called during `clone`.
   */
  public AdditiveModel generateWeightVector() {
    for (Map.Entry<String, Map<String, Integer>> family : featureIndexer.entrySet()) {
      String familyName = family.getKey();
      for (Map.Entry<String, Integer> feature : family.getValue().entrySet()) {
        weightVector[feature.getValue()] = weights.get(familyName).get(feature.getKey());
      }
    }
    return this;
  }

  @Override
  public float scoreItem(FeatureVector combinedItem) {
    Map<String, Map<String, Double>> flatFeatures = Util.flattenFeature(combinedItem);
    return scoreFlatFeatures(flatFeatures) + scoreDenseFeatures(combinedItem.getDenseFeatures());
  }

  public float scoreDenseFeatures(Map<String, List<Double>> denseFeatures) {
    float sum = 0;
    if (denseFeatures != null && !denseFeatures.isEmpty()) {
      Map<String, Function> denseWeights = getOrCreateDenseWeights();
      for (Map.Entry<String, List<Double>> feature : denseFeatures.entrySet()) {
        String featureName = feature.getKey();
        Function fun = denseWeights.get(featureName);
        if (fun == null) continue;
        sum += fun.evaluate(toFloat(feature.getValue()));
      }
    }
    return sum;
  }

  @Override
  public float debugScoreItem(FeatureVector combinedItem,
                              StringBuilder builder) {
    Map<String, Map<String, Double>> flatFeatures = Util.flattenFeature(combinedItem);

    float sum = 0.0f;
    // order by the absolute value
    PriorityQueue<Map.Entry<String, Float>> scores =
        new PriorityQueue<>(100, new LinearModel.EntryComparator());

    for (Map.Entry<String, Map<String, Double>> featureFamily : flatFeatures.entrySet()) {
      Map<String, Function> familyWeightMap = weights.get(featureFamily.getKey());
      if (familyWeightMap == null)
        continue;
      for (Map.Entry<String, Double> feature : featureFamily.getValue().entrySet()) {
        Function func = familyWeightMap.get(feature.getKey());
        if (func == null)
          continue;
        float val = feature.getValue().floatValue();
        float subScore = func.evaluate(val);
        sum += subScore;
        String str = featureFamily.getKey() + ":" + feature.getKey() + "=" + val
            + " = " + subScore + "<br>\n";
        scores.add(new AbstractMap.SimpleEntry<>(str, subScore));
      }
    }

    Map<String, List<Double>> denseFeatures = combinedItem.getDenseFeatures();
    if (denseFeatures != null) {
      assert (denseWeights != null);
      for (Map.Entry<String, List<Double>> feature : denseFeatures.entrySet()) {
        String featureName = feature.getKey();
        Function fun = denseWeights.get(featureName);
        float[] val = toFloat(feature.getValue());
        float subScore = fun.evaluate(val);
        sum += subScore;
        String str = DENSE_FAMILY + ":" + featureName + "=" + val
            + " = " + subScore + "<br>\n";
        scores.add(new AbstractMap.SimpleEntry<>(str, subScore));
      }
    }

    final int MAX_COUNT = 100;
    builder.append("Top scores ===>\n");
    if (!scores.isEmpty()) {
      int count = 0;
      float subsum = 0.0f;
      while (!scores.isEmpty()) {
        Map.Entry<String, Float> entry = scores.poll();
        builder.append(entry.getKey());
        float val = entry.getValue();
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
    Map<String, Map<String, Double>> flatFeatures = Util.flattenFeature(combinedItem);
    List<DebugScoreRecord> scoreRecordsList = new ArrayList<>();

    for (Map.Entry<String, Map<String, Double>> featureFamily : flatFeatures.entrySet()) {
      Map<String, Function> familyWeightMap = weights.get(featureFamily.getKey());
      if (familyWeightMap == null) continue;
      for (Map.Entry<String, Double> feature : featureFamily.getValue().entrySet()) {
        Function func = familyWeightMap.get(feature.getKey());
        if (func == null) continue;
        float val = feature.getValue().floatValue();
        float weight = func.evaluate(val);
        DebugScoreRecord record = new DebugScoreRecord();
        record.setFeatureFamily(featureFamily.getKey());
        record.setFeatureName(feature.getKey());
        record.setFeatureValue(val);
        record.setFeatureWeight(weight);
        scoreRecordsList.add(record);
      }
    }

    Map<String, List<Double>> denseFeatures = combinedItem.getDenseFeatures();
    if (denseFeatures != null) {
      Map<String, Function> denseWeights = getOrCreateDenseWeights();
      for (Map.Entry<String, List<Double>> feature : denseFeatures.entrySet()) {
        String featureName = feature.getKey();
        Function fun = denseWeights.get(featureName);
        float[] val = toFloat(feature.getValue());
        float weight = fun.evaluate(val);
        DebugScoreRecord record = new DebugScoreRecord();
        record.setFeatureFamily(DENSE_FAMILY);
        record.setFeatureName(feature.getKey());
        record.setDenseFeatureValue(feature.getValue());
        record.setFeatureWeight(weight);
        scoreRecordsList.add(record);
      }
    }

    return scoreRecordsList;
  }

  @Override
  protected void loadInternal(ModelHeader header, BufferedReader reader) throws IOException {
    long rows = header.getNumRecords();
    slope = header.getSlope();
    offset = header.getOffset();
    weights = new HashMap<>();
    for (long i = 0; i < rows; i++) {
      String line = reader.readLine();
      ModelRecord record = Util.decodeModel(line);
      String family = record.getFeatureFamily();
      String name = record.getFeatureName();
      Map<String, Function> inner = weights.get(family);
      if (inner == null) {
        inner = new HashMap<>();
        weights.put(family, inner);
      }
      inner.put(name, AbstractFunction.buildFunction(record));
    }
  }

  @Override
  public void save(BufferedWriter writer) throws IOException {
    ModelHeader header = new ModelHeader();
    header.setModelType("additive");
    header.setSlope(slope);
    header.setOffset(offset);
    long count = 0;
    for (Map.Entry<String, Map<String, Function>> familyMap : weights.entrySet()) {
      count += familyMap.getValue().size();
    }
    header.setNumRecords(count);
    ModelRecord headerRec = new ModelRecord();
    headerRec.setModelHeader(header);
    writer.write(Util.encode(headerRec));
    writer.newLine();
    for (Map.Entry<String, Map<String, Function>> familyMap : weights.entrySet()) {
      String featureFamily = familyMap.getKey();
      for (Map.Entry<String, Function> feature : familyMap.getValue().entrySet()) {
        Function func = feature.getValue();
        String featureName = feature.getKey();
        writer.write(Util.encode(func.toModelRecord(featureFamily, featureName)));
        writer.newLine();
      }
    }
    writer.flush();
  }

  public float scoreFlatFeatures(Map<String, Map<String, Double>> flatFeatures) {
    float sum = 0.0f;
    for (Map.Entry<String, Map<String, Double>> featureFamily : flatFeatures.entrySet()) {
      Map<String, Function> familyWeightMap = weights.get(featureFamily.getKey());
      if (familyWeightMap == null) {
        // not important families/features are removed from model
        log.debug("miss featureFamily {}", featureFamily.getKey());
        continue;
      }
      for (Map.Entry<String, Double> feature : featureFamily.getValue().entrySet()) {
        Function func = familyWeightMap.get(feature.getKey());
        if (func == null)
          continue;
        float val = feature.getValue().floatValue();
        sum += func.evaluate(val);
      }
    }
    return sum;
  }

  public float scoreFeatures(SparseLabeledPoint point, double dropout, Random rand) {
    float prediction = 0;

    for (int i = 0; i < point.indices.length; i++) {
      if (dropout <= 0 || rand.nextDouble() > dropout) {
        int index = point.indices[i];
        Function function = weightVector[index];
        float value = point.values[i];
        prediction += function.evaluate(value);
      }
    }

    for (int i = 0; i < point.denseIndices.length; i++) {
      if (dropout <= 0 || rand.nextDouble() > dropout) {
        int index = point.denseIndices[i];
        Function function = weightVector[index];
        float[] value = point.denseValues[i];
        prediction += function.evaluate(value);
      }
    }

    return prediction;
  }

  public Map<String, Function> getOrCreateFeatureFamily(String featureFamily) {
    Map<String, Function> featFamily = weights.get(featureFamily);
    if (featFamily == null) {
      featFamily = new HashMap<>();
      weights.put(featureFamily, featFamily);
    }
    return featFamily;
  }

  public void addFunction(String featureFamily, String featureName,
                          Function function, boolean overwrite) {
    if (function == null) {
      throw new RuntimeException(featureFamily + " " + featureName + " function null");
    }
    Map<String, Function> featFamily = getOrCreateFeatureFamily(featureFamily);
    if (overwrite || !featFamily.containsKey(featureName)) {
      featFamily.put(featureName, function);
    }
  }

  public void update(float gradWithLearningRate, SparseLabeledPoint point, double dropout, Random rand) {
    for (int i = 0; i < point.indices.length; i++) {
      if (dropout <= 0 || rand.nextDouble() > dropout) {
        int index = point.indices[i];
        Function function = weightVector[index];
        float value = point.values[i];
        function.update(-gradWithLearningRate, value);
      }
    }

    for (int i = 0; i < point.denseIndices.length; i++) {
      if (dropout <= 0 || rand.nextDouble() > dropout) {
        int index = point.denseIndices[i];
        Function function = weightVector[index];
        float[] value = point.denseValues[i];
        function.update(-gradWithLearningRate, value);
      }
    }
  }

  @Override
  public AdditiveModel clone() throws CloneNotSupportedException {
    AdditiveModel copy = (AdditiveModel) super.clone();

    // deep copy weights
    Map<String, Map<String, Function>> newWeights = new HashMap<>();
    weights.forEach((k, v) -> newWeights.put(k, copyFeatures(v)));
    copy.weights = newWeights;

    copy.denseWeights = copyFeatures(denseWeights);

    // regenerate weight vector
    copy.weightVector = new Function[copy.weightVector.length];
    return copy.generateWeightVector();
  }

  private Map<String, Function> copyFeatures(Map<String, Function> featureMap) {
    if (featureMap == null) return null;

    Map<String, Function> newFeatureMap = new HashMap<>();
    featureMap.forEach((feature, function) -> {
      try {
        newFeatureMap.put(feature, function.clone());
      } catch (CloneNotSupportedException e) {
        // Java8 stream does not handle checked exception properly and requires explicit handling
        e.printStackTrace();
      }
    });
    return newFeatureMap;
  }
}
