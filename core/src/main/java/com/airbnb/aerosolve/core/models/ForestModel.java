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
import java.util.Random;
import java.util.Set;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ForestModelOptions;
import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.util.Util;
import com.airbnb.aerosolve.core.util.Spline;

import lombok.Getter;
import lombok.Setter;

// A tree forest model.
public class ForestModel extends AbstractModel {

  private static final long serialVersionUID = 3651061358422885378L;

  @Getter @Setter
  protected ArrayList<DecisionTreeModel> trees;

  // For online learning.
  class ForestSample {
    public float update;
    public Map<String, Map<String, Double>> flatFeatures;
    ForestSample(float up, Map<String, Map<String, Double>> ff) {
      update = up;
      flatFeatures = ff;
    }
  }
  
  @Getter @Setter
  protected ForestModelOptions options;
  protected ArrayList<ForestSample> samples;

  public ForestModel() {
    trees = new ArrayList<>();
  }

  @Override
  public float scoreItem(FeatureVector combinedItem) {
    Map<String, Map<String, Double>> floatFeatures = Util.flattenFeature(combinedItem);

    float sum = 0.0f;
    // Note: we sum instead of average so that the trainer has the option of boosting the
    // trees together.
    for (int i = 0; i < trees.size(); i++) {
      sum += trees.get(i).scoreFlattenedFeature(floatFeatures);
    }
    return sum;
  }

  @Override
  // Forests don't usually have debuggable components.
  public float debugScoreItem(FeatureVector combinedItem,
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
      DecisionTreeModel tree = new DecisionTreeModel();
      tree.loadInternal(record.getModelHeader(), reader);
      trees.add(tree);
    }
  }
  
  // Mostly for reinforcement learning, it differs from the batch trainer in that it learns the delta from
  // the previous models.
  @Override
  public void onlineUpdate(float grad, float learningRate, Map<String, Map<String, Double>> flatFeatures) {
    // Create array list if it is not there.
    if (samples == null) {
      samples = new ArrayList<>();
    }
    // Use defaults if options were not set.
    if (options == null) {
      options = new ForestModelOptions();
    }
    samples.add(new ForestSample(-learningRate * grad, flatFeatures));
    if (samples.size() >= options.sampleSize) {
      addNewTree(options, samples);
      samples = null;
    }
  }
  
  private void addNewTree(ForestModelOptions options, ArrayList<ForestSample> samples) {
    ArrayList<ModelRecord> stumps = new ArrayList<>();
    stumps.add(new ModelRecord());
    buildTree(stumps, samples, 0, 0, options);
    DecisionTreeModel tree = new DecisionTreeModel();
    tree.setStumps(stumps);
    trees.add(tree);
    /*
    System.out.println("Tree " + trees.size());
    for (ModelRecord rec : stumps) {
      System.out.println(rec.toString());
    }*/
  }
  
  // A variant of the scala function but just built for regression for adding new trees to the forest.
  public static void buildTree(
      ArrayList<ModelRecord> stumps,
      ArrayList<ForestSample> examples,
      int currIdx,
      int currDepth,
      ForestModelOptions options) {
    if (currDepth >= options.maxDepth) {
      stumps.set(currIdx, makeLeaf(examples));
      return;
    }
    ModelRecord split = getBestSplit(examples, options);
    
    if (split == null) {
      stumps.set(currIdx, makeLeaf(examples));
      return;
    }
    
    //This is a split node.
    stumps.set(currIdx, split);    
    int left = stumps.size();
    stumps.add(new ModelRecord());
    int right = stumps.size();
    stumps.add(new ModelRecord());
    stumps.get(currIdx).setLeftChild(left);
    stumps.get(currIdx).setRightChild(right);
    
    ArrayList<ForestSample> leftEx = new ArrayList<>();
    ArrayList<ForestSample> rightEx = new ArrayList<>();
    for (ForestSample ex : examples) {
      if (BoostedStumpsModel.getStumpResponse(stumps.get(currIdx), ex.flatFeatures)) {
        rightEx.add(ex);
      } else {
        leftEx.add(ex);
      }
    }
    
    buildTree(stumps, leftEx, left, currDepth + 1, options);
    buildTree(stumps, rightEx, right, currDepth + 1, options);
  }
  
  private static ModelRecord makeLeaf(ArrayList<ForestSample> examples) {
    ModelRecord rec = new ModelRecord();
    double count = 0.0;
    double sum = 0.0;
    
    for (ForestSample example : examples) {
      count += 1.0;
      sum += example.update;
    }
    rec.setFeatureWeight(sum / count);
    return rec;
  }
  
  private static ModelRecord getCandidateSplit(ForestSample ex, Random rnd) {
    // Flatten the features and pick one randomly.
    ArrayList<String> families = new ArrayList<String>();
    for (String family : ex.flatFeatures.keySet()) {
      families.add(family);
    }
    if (families.isEmpty()) {
      return null;
    }
    int familyIdx = rnd.nextInt(families.size());
    String family = families.get(familyIdx);
    ArrayList<String> values = new ArrayList<String>();
    Map<String, Double> familyMap = ex.flatFeatures.get(family);
    for (String valname : familyMap.keySet()) {
      values.add(valname);
    }
    if (values.isEmpty()) {
      return null;
    }    
    int valueIdx = rnd.nextInt(values.size());
    String valueName = values.get(valueIdx);
    Double value = familyMap.get(valueName);
    ModelRecord rec = new ModelRecord();
    rec.setFeatureFamily(family);
    rec.setFeatureName(valueName);
    rec.setThreshold(value);
    return rec;
  }
  
  private static ModelRecord getBestSplit(ArrayList<ForestSample> examples, ForestModelOptions options) {
   ModelRecord bestRecord = null;
   // Find the splits of minimum variance.
   double bestValue = 1e10;
   Random rnd = new Random(1234);
   for (int i = 0; i < options.getNumTries(); i++) {
     // Pick an example index randomly
     int idx = rnd.nextInt(examples.size());
     ForestSample ex = examples.get(idx);
     ModelRecord candidate = getCandidateSplit(ex, rnd);
     if (candidate != null) {
       double candidateValue = evaluateRegressionSplit(examples, candidate, options);
       if (candidateValue < bestValue) {
         bestValue = candidateValue;
         bestRecord = candidate;
       }
     }
   }
   
   return bestRecord;
 }
  
  private static double evaluateRegressionSplit(ArrayList<ForestSample> examples,
                                                ModelRecord candidate,
                                                ForestModelOptions options) {
    double rightCount, rightMean, rightSumSq, leftCount, leftMean, leftSumSq;
    rightCount = rightMean = rightSumSq = leftCount = leftMean = leftSumSq = 0.0;

    for (ForestSample example : examples) {
      boolean response = BoostedStumpsModel.getStumpResponse(candidate, example.flatFeatures);
      double labelValue = example.update;

      // Using Welford's Method for computing mean and sum-squared errors in numerically stable way;
      // more details can be found in http://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance
      //
      // See unit test for verification that it is consistent with standard, two-pass approach
      if (response) {
        rightCount += 1;
        double delta = labelValue - rightMean;
        rightMean += delta / rightCount;
        rightSumSq += delta * (labelValue - rightMean);
      } else {
        leftCount += 1;
        double delta = labelValue - leftMean;
        leftMean += delta / leftCount;
        leftSumSq += delta * (labelValue - leftMean);
      }
    }

    if (rightCount >= options.minItemCount && leftCount >= options.minItemCount) {
      return leftSumSq + rightSumSq;
    }
    return 1e10;
  }
}
