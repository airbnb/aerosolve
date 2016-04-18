package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.airbnb.aerosolve.core.models.KDTreeModel;
import com.airbnb.aerosolve.core.KDTreeNode;
import com.google.common.base.Optional;
import com.typesafe.config.Config;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Inputs = fieldName1 (value1, value2)
 * Outputs = list of kdtree nodes and the distance from the split
 * This is the continuous version of the kd-tree transform and encodes
 * the distance from each splitting plane to the point being queried.
 * One can think of this as a tree kernel transform of a point.
 */
public class KdtreeContinuousTransform implements Transform {
  private String fieldName1;
  private String value1;
  private String value2;
  private String outputName;
  private Integer maxCount;
  private Optional<KDTreeModel> modelOptional;
  private static final Logger log = LoggerFactory.getLogger(KdtreeContinuousTransform.class);

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    value1 = config.getString(key + ".value1");
    value2 = config.getString(key + ".value2");
    outputName = config.getString(key + ".output");
    maxCount = config.getInt(key + ".max_count");
    String modelEncoded = config.getString(key + ".model_base64");

    modelOptional = KDTreeModel.readFromGzippedBase64String(modelEncoded);

    if (!modelOptional.isPresent()) {
      log.error("Could not load KDTree from encoded field");
    }
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    if (!modelOptional.isPresent()) {
      return;
    }
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();

    if (floatFeatures == null) {
      return;
    }

    Map<String, Double> feature1 = floatFeatures.get(fieldName1);
    if (feature1 == null) {
      return;
    }

    Double v1 = feature1.get(value1);
    Double v2 = feature1.get(value2);

    if (v1 == null || v2 == null) {
      return;
    }

    Map<String, Double> output = Util.getOrCreateFloatFeature(outputName, floatFeatures);

    ArrayList<Integer> result = modelOptional.get().query(v1, v2);
    int count = Math.min(result.size(), maxCount);
    KDTreeNode[] nodes = modelOptional.get().getNodes();

    for (int i = 0; i < count; i++) {
      Integer res = result.get(result.size() - 1 - i);
      double split = nodes[res].getSplitValue();
      switch (nodes[res].getNodeType()) {
        case X_SPLIT: {
          output.put(res.toString(), v1 - split);
        }
        break;
        case Y_SPLIT: {
          output.put(res.toString(), v2 - split);
        }
        break;
      }
    }
  }
}
