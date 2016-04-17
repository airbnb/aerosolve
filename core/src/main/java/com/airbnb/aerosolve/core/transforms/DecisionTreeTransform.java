package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.models.DecisionTreeModel;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.io.Serializable;
import java.util.*;
import java.util.Map.Entry;

/**
 * Applies a decision tree transform to existing float features.
 * Emits the binary leaf features to the string family output_leaves
 * Emits the score to the float family output_score
 * Use tree.toHumanReadableTransform to generate the nodes list.
 */
public class DecisionTreeTransform implements Transform {
  private String outputLeaves;
  private String outputScoreFamily;
  private String outputScoreName;

  private DecisionTreeModel tree;

  @Override
  public void configure(Config config, String key) {
    outputLeaves = config.getString(key + ".output_leaves");
    outputScoreFamily = config.getString(key + ".output_score_family");
    outputScoreName = config.getString(key + ".output_score_name");
    List<String> nodes = config.getStringList(key + ".nodes");
    tree = DecisionTreeModel.fromHumanReadableTransform(nodes);
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();

    if (floatFeatures == null) {
      return;
    }

    Util.optionallyCreateStringFeatures(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    Set<String> outputString = Util.getOrCreateStringFeature(outputLeaves, stringFeatures);
    
    Map<String, Double> outputFloat = Util.getOrCreateFloatFeature(outputScoreFamily, floatFeatures);
    int leafIdx = tree.getLeafIndex(floatFeatures);
    ModelRecord rec = tree.getStumps().get(leafIdx);
    outputString.add(rec.featureName);
    outputFloat.put(outputScoreName, rec.featureWeight);
  }
}
