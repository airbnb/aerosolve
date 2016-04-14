package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.features.DenseVector;
import com.airbnb.aerosolve.core.features.FamilyVector;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;

public class FeatureVectorUtil {
  /**
   * Computes the min kernel between two double arrays.
   * @param a - first double array
   * @param b - second double array
   * @return - sum(min(a(i), b(i))
   */
  public static double minKernel(double[] a, double[] b) {
    double sum = 0.0;
    if (a == null || b == null) {
      return 0.0;
    }

    // This stops at the shorter array. Since we're taking the min, we can assume the shorter array
    // could be interpreted as 0 beyond it's length and that would be less than b.  Is this true?
    // what if the other array has negatives?
    for (int i = 0; i < Math.min(a.length, b.length); i++) {
      sum += Math.min(a[i], b[i]);
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
    if (a instanceof MultiFamilyVector && b instanceof MultiFamilyVector) {
      double sum = 0.0;

      MultiFamilyVector multiB = (MultiFamilyVector) b;
      for (FamilyVector vec : ((MultiFamilyVector) a).families()) {
        sum += denseVectorMinKernel(vec, multiB.get(vec.family()));
      }
      return sum;
    }
    return denseVectorMinKernel(a, b);
  }

  private static double denseVectorMinKernel(FeatureVector a,
                                            FeatureVector b) {
    if (a instanceof DenseVector && b instanceof DenseVector) {
      return minKernel(a.denseArray(), b.denseArray());
    }
    return 0.0;
  }
}