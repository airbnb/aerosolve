package com.airbnb.aerosolve.core.function;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;

import java.util.Arrays;
import java.util.List;

public class FunctionUtil {
  public static double[] fitPolynomial(double[] data) {
    int numCoeff = 6;
    int iterations = numCoeff * 4;
    double[] initial = new double[numCoeff];

    double[] initialStep = new double[numCoeff];
    Arrays.fill(initialStep, 1.0d);
    return optimize(1.0 / 512.0, iterations, initial, initialStep,
        new ImmutablePair<Double, Double>(-10.0d, 10.0d), data);
  }

  public static double evaluatePolynomial(double[] coeff, double[] data, boolean overwrite) {
    int len = data.length;
    double err = 0;
    long count = 0;
    for (int i = 0; i < len; i++) {
      double t = (double) i / (len - 1);
      double tinv = 1 - t;
      double diracStart = (i == 0) ? coeff[0] : 0;
      double diracEnd = (i == len - 1) ? coeff[1] : 0;
      double eval = coeff[2] * tinv * tinv * tinv +
          coeff[3] * 3.0 * tinv * tinv * t +
          coeff[4] * 3.0 * tinv * t * t +
          coeff[5] * t * t * t +
          diracStart +
          diracEnd;
      if (data[i] != 0.0) {
        err += Math.abs(eval - data[i]);
        count++;
      }
      if (overwrite) {
        data[i] = eval;
      }
    }
    return err / count;
  }

  // CyclicCoordinateDescent
  public static double[] optimize(double tolerance, int iterations,
                                 double[] initial, double[] initialStep,
                                 Pair<Double, Double> bounds, double[] data) {
    double[] best = initial;
    double bestF = evaluatePolynomial(best, data, false);
    int maxDim = initial.length;
    for (int i = 0; i < iterations; ++i) {
      for (int dim = 0; dim < maxDim; ++dim) {
        double step = initialStep[dim];
        while (step > tolerance) {
          double[] left = best.clone();
          left[dim] = Math.max(bounds.getLeft(), best[dim] - step);
          double leftF = evaluatePolynomial(left, data, false);
          double[] right = best.clone();
          right[dim] = Math.min(bounds.getRight(), best[dim] + step);
          double rightF = evaluatePolynomial(right, data, false);
          if (leftF < bestF) {
            best = left;
            bestF = leftF;
          }
          if (rightF < bestF) {
            best = right;
            bestF = rightF;
          }
          step *= 0.5;
        }
      }
    }
    return best;
  }

  /*
 * @param  tolerance if fitted array's deviation from weights is less than tolerance
 *         use the fitted, otherwise keep original weights.
 * @param  weights the curve you want to smooth
 * @return true if weights is modified by fitted curve.
   */
  public static boolean smooth(double tolerance, double[] weights) {
    // TODO use apache math's PolynomialCurveFitter
    // compile 'org.apache.commons:commons-math3:3.6.1'
    double[] best = FunctionUtil.fitPolynomial(weights);
    double errAndCoeff = FunctionUtil.evaluatePolynomial(best, weights, false);
    if (errAndCoeff < tolerance) {
      FunctionUtil.evaluatePolynomial(best, weights, true);
      return true;
    } else {
      return false;
    }
  }
}
