package com.airbnb.aerosolve.core.transforms.types;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.transforms.Transform;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.util.Map;

public abstract class FloatTransform implements Transform {
  protected String fieldName1;
  protected String outputName; // output family name, if not specified, output to fieldName1

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    if (config.hasPath(key + ".output")) {
      outputName = config.getString(key + ".output");
    } else {
      outputName = fieldName1;
    }
    init(config, key);
  }
  protected abstract void init(Config config, String key);

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();

    if (floatFeatures == null) {
      return;
    }
    Map<String, Double> input = floatFeatures.get(fieldName1);
    if (input == null) {
      return;
    }
    Map<String, Double> output = Util.getOrCreateFloatFeature(outputName, floatFeatures);
    output(input, output);
  }

  protected abstract void output(Map<String, Double> input, Map<String, Double> output);
}
