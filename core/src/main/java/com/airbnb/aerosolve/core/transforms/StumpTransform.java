package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.io.Serializable;
import java.util.*;
import java.util.Map.Entry;

/**
 * Applies boosted stump transform to float features and places them in string feature output.
 * The format for a stump feature family, feature name, threshold, descriptive name
 * You can obtain the stumps from a BoostedStump model in spark shell using
 * val model = sc.textFile(name).map(Util.decodeModel).take(10).map(x =&gt;
 *  "%s,%s,%f".format(x.featureFamily,x.featureName,x.threshold)).foreach(println)
 */
public class StumpTransform implements Transform {
  private String outputName;

  private class StumpDescription implements Serializable {
    public StumpDescription(String featureName, Double threshold, String descriptiveName) {
      this.featureName = featureName;
      this.threshold = threshold;
      this.descriptiveName = descriptiveName;
    }
    public String featureName;
    public Double threshold;
    public String descriptiveName;
  }

  // Family -> description
  private Map<String, List<StumpDescription>> thresholds;

  @Override
  public void configure(Config config, String key) {
    outputName = config.getString(key + ".output");
    thresholds = new HashMap<>();

    List<String> stumps = config.getStringList(key + ".stumps");
    for (String stump : stumps) {
      String[] tokens = stump.split(",");
      if (tokens.length == 4) {
        String family = tokens[0];
        String featureName = tokens[1];
        Double threshold = Double.parseDouble(tokens[2]);
        String descriptiveName = tokens[3];
        List<StumpDescription> featureList = thresholds.get(family);
        if (featureList == null) {
          featureList = new ArrayList<>();
          thresholds.put(family, featureList);
        }
        StumpDescription description = new StumpDescription(featureName,
                                                            threshold,
                                                            descriptiveName);
        featureList.add(description);
      }
    }
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();

    if (floatFeatures == null) {
      return;
    }

    Util.optionallyCreateStringFeatures(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    Set<String> output = Util.getOrCreateStringFeature(outputName, stringFeatures);

    for (Entry<String, List<StumpDescription>> stumpFamily : thresholds.entrySet()) {
      Map<String, Double> feature = floatFeatures.get(stumpFamily.getKey());
      if (feature == null) continue;
      for (StumpDescription desc : stumpFamily.getValue()) {
        Double value = feature.get(desc.featureName);
        if (value != null && value >= desc.threshold) {
          output.add(desc.descriptiveName);
        }
      }
    }
  }
}
