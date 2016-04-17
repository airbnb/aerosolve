package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/*
  Turn several float features into one dense feature, feature number must > 1
  1. IF all float features are null, create a string feature,
      with family name string_output, feature name output^null
  2. IF only one float feature is not null, create a float feature
     with family name same as family of the only not null float feature
  3. Other cases create dense features
   both 2 and 3, feature name: output^key keys.
 */
public class FloatToDenseTransform implements Transform{
  private List<String> fields;
  private List<String> keys;
  private String outputName;
  private String outputStringFamily;
  private static final int featureAVGSize = 16;
  @Override
  public void configure(Config config, String key) {
    outputStringFamily = config.getString(key + ".string_output");
    outputName = config.getString(key + ".output");
    keys = config.getStringList(key + ".keys");
    fields = config.getStringList(key + ".fields");
    if (fields.size() != keys.size() || fields.size() <= 1) {
      String msg = String.format("fields size {} keys size {}", fields.size(), keys.size());
      throw new RuntimeException(msg);
    }
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();

    if (floatFeatures == null) {
      return;
    }
    int size = fields.size();
    StringBuilder sb = new StringBuilder((size + 1) * featureAVGSize);
    sb.append(outputName);
    List<Double> output = new ArrayList<>(size);
    Map<String, Double> floatFamily = null;
    for (int i = 0; i < size; ++i) {
      String familyName = fields.get(i);
      Map<String, Double> family = floatFeatures.get(familyName);
      if (family == null) {
        continue;
      }
      String featureName = keys.get(i);
      Double feature = family.get(keys.get(i));
      if (feature != null) {
        output.add(feature);
        sb.append('^');
        sb.append(featureName);
        floatFamily = family;
      }
    }

    switch (output.size()) {
      case 0: {
        sb.append("^null");
        Util.setStringFeature(featureVector, outputStringFamily, sb.toString());
      }
      break;
      case 1: {
        floatFamily.put(sb.toString(), output.get(0));
      }
      break;
      default:
        Util.setDenseFeature(featureVector, sb.toString(), output);
    }
  }
}
