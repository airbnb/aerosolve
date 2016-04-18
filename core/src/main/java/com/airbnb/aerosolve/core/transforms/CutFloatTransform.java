package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;

import java.util.List;
import java.util.Map;

/*
  remove features larger than upperBound or smaller than lowerBound
 */
public class CutFloatTransform implements Transform {
    private String fieldName1;
    private List<String> keys;
    private double lowerBound;
    private double upperBound;
    private String outputName; // output family name, if not specified, output to fieldName1

    @Override
    public void configure(Config config, String key) {
        fieldName1 = config.getString(key + ".field1");
        keys = config.getStringList(key + ".keys");
        if (config.hasPath(key + ".lower_bound")) {
            lowerBound = config.getDouble(key + ".lower_bound");
        } else {
            lowerBound = -Double.MAX_VALUE;
        }
        if (config.hasPath(key + ".upper_bound")) {
            upperBound = config.getDouble(key + ".upper_bound");
        } else {
            upperBound = Double.MAX_VALUE;
        }
        if (config.hasPath(key + ".output")) {
            outputName = config.getString(key + ".output");
        } else {
            outputName = fieldName1;
        }
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
        Map<String, Double> feature2 = Util.getOrCreateFloatFeature(outputName, floatFeatures);
        for (String key : keys) {
            Double v = feature1.get(key);
            if (v != null) {
                if (v > upperBound || v < lowerBound) {
                    if (feature2 == feature1) {
                        feature2.remove(key);
                    }
                } else if (feature2 != feature1) {
                    feature2.put(key, v);
                }
            }
        }
    }
}
