package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class DeleteStringFeatureTransform implements Transform {
  private String fieldName1;
  private List<String> keys;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    keys = config.getStringList(key + ".keys");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();

    if (stringFeatures == null) {
      return;
    }

    Set<String> feature1 = stringFeatures.get(fieldName1);
    if (feature1 == null) {
      return;
    }

    List<String> toDelete = new ArrayList<String>();
    for(String feat : feature1) {
      for (String key : keys) {
        if (feat.startsWith(key)) {
          toDelete.add(feat);
          break;
        }
      }
    }
    for (String feat : toDelete) {
      feature1.remove(feat);
    }
  }
}
