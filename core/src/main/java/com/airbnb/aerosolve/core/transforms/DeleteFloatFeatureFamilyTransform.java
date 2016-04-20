package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;

import java.util.HashSet;
import java.util.List;
import java.util.Map;

import com.typesafe.config.Config;

/**
 * "field1" optionally specifies the float feature family to be deleted
 * "fields" optionally specifies a list of float feature families to be deleted
 */
public class DeleteFloatFeatureFamilyTransform implements Transform {
  private String fieldName1;
  private List<String> fieldNames;

  @Override
  public void configure(Config config, String key) {
    if (config.hasPath(key + ".field1")) {
      fieldName1 = config.getString(key + ".field1");
    }
    if (config.hasPath(key + ".fields")) {
      fieldNames = config.getStringList(key + ".fields");
    }
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    if (floatFeatures == null) {
      return ;
    }

    HashSet<String> fieldNamesSet = new HashSet<>();

    if (fieldName1 != null) {
      fieldNamesSet.add(fieldName1);
    }
    if (fieldNames != null) {
      fieldNamesSet.addAll(fieldNames);
    }

    for (String fieldName: fieldNamesSet) {
      Map<String, Double> feature = floatFeatures.get(fieldName);
      if (feature != null) {
        floatFeatures.remove(fieldName);
      }
    }
  }
}
