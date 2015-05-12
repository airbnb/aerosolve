package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.FeatureVector;

import java.util.List;
import java.util.Set;

public class FeatureVectorUtil {
  /**
   * Computes the min kernel for one feature family.
   * @param featureKey - name of feature e.g. "rgb"
   * @param a - first feature vector
   * @param b - second feature vector
   * @return - sum(min(a(i), b(i))
   */
  public static double featureMinKernel(String featureKey,
                                        FeatureVector a,
                                        FeatureVector b) {
    double sum = 0.0;
    if (a.getDenseFeatures() == null || b.getDenseFeatures() == null) {
      return 0.0;
    }
    List<Double> aFeat = a.getDenseFeatures().get(featureKey);
    List<Double> bFeat = b.getDenseFeatures().get(featureKey);
    if (aFeat == null || bFeat == null) {
      return 0.0;
    }
    int count = aFeat.size();
    for (int i = 0; i < count; i++) {
      if (aFeat.get(i) < bFeat.get(i)) {
        sum += aFeat.get(i);
      } else {
        sum += bFeat.get(i);
      }
    }
    return sum;
  }

  /**
   *
   * @param a - a feature vector
   * @param b - another feature vector
   * @return the min kernel between both feature vectors.
   */
  public static double featureVectorMinKernel(FeatureVector a,
                                              FeatureVector b) {
    double sum = 0.0;
    if (a.getDenseFeatures() == null) {
      return 0.0;
    }
    Set<String> keys = a.getDenseFeatures().keySet();
    for (String key : keys) {
      sum += featureMinKernel(key, a, b);
    }
    return sum;
  }
}