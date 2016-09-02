package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;

import java.util.Map;

import com.typesafe.config.Config;

/**
 * Convert a Float feature to binary LABEL based on threshold value. This is intended for
 * binary classification where LABEL is either -1 or 1. MergeStrategy describe how existing
 * LABEL is treated.
 *
 * The purpose of MergeStrategy is to help combining float features into LABEL. For example,
 * using OVERRIDE_NEGATIVE is equivalent to LABEL with feature1 >= threshold1 or feature2 >= threshold2,
 * using OVERRIDE_POSITIVE is equivalent to LABEL with feature1 >= threshold1 and feature2 >= threshold2.
 */
public class FloatLabelTransform implements Transform {
  enum MergeStrategy {
    OVERRIDE, // override existing label
    OVERRIDE_NEGATIVE, // override negative label but keep positive label
    OVERRIDE_POSITIVE, // override positive label but keep negative label
    SKIP // preserve existing label (only replace if observation is un-labeled)
  }

  private String fieldName1;
  private String key1;
  private double threshold;
  private MergeStrategy mergeStrategy;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    key1 = config.getString(key + ".key1");
    threshold = config.getDouble(key + ".threshold");
    mergeStrategy = MergeStrategy.valueOf(config.getString(key + ".merge").toUpperCase());
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Double> labelFeature = Util.getOrCreateFloatFeature("LABEL", featureVector.floatFeatures);

    Double label = labelFeature.get("");

    if (mergeStrategy == MergeStrategy.SKIP && label != null) return;

    Map<String, Double> floatFeature = featureVector.floatFeatures.get(fieldName1);
    if (floatFeature != null) {
      Double featureValue = floatFeature.get(key1);
      if (featureValue != null) {
        double newLabel = featureValue >= threshold ? 1 : -1;

        if (
          // fill in missing label
            label == null ||
                // ignore existing label
                mergeStrategy == MergeStrategy.OVERRIDE ||
                // override negative label
                (mergeStrategy == MergeStrategy.OVERRIDE_NEGATIVE && label < 0 && newLabel > 0) ||
                // override positive label
                (mergeStrategy == MergeStrategy.OVERRIDE_POSITIVE && label > 0 && newLabel < 0)
            ) {
          labelFeature.put("", newLabel);
        }
      }
    }
  }
}
