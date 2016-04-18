package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.util.Map;

/**
 * id = fieldName1.key1
 * feature value = fieldName2.key2
 * output[ fieldname2 @ id ] = feature value
 * This transform is useful for making cross products of categorical features
 * e.g. leaf_id (say 123) and a continuous variable e.g. searches_at_leaf (say 4.0)
 * and making a new feature searches_at_leaf @ 123 = 4.0
 * The original searches_at_leaf feature can compare quantities at a global level
 * say searches in one market vs another market.
 * On the other hand searches_at_leaf @ 123 can tell you how the model changes
 * for searches at a particular place changing from day to day.
 */
public class StuffIdTransform implements Transform {
  private String fieldName1;
  private String fieldName2;
  private String key1;
  private String key2;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    fieldName2 = config.getString(key + ".field2");
    key1 = config.getString(key + ".key1");
    key2 = config.getString(key + ".key2");
    outputName = config.getString(key + ".output");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();

    if (floatFeatures == null) {
      return;
    }

    Map<String, Double> feature1 = floatFeatures.get(fieldName1);
    if (feature1 == null) {
      return;
    }

    Map<String, Double> feature2 = floatFeatures.get(fieldName2);
    if (feature2 == null) {
      return;
    }

    Double v1 = feature1.get(key1);
    Double v2 = feature2.get(key2);
    if (v1 == null || v2 == null) {
      return;
    }

    Map<String, Double> output = Util.getOrCreateFloatFeature(outputName, floatFeatures);

    String newname = key2 + '@' + v1.longValue();
    output.put(newname, v2);
  }
}
