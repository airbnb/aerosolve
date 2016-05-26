package com.airbnb.aerosolve.core.function;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;

import java.util.Arrays;
import java.util.List;

@Slf4j
public class FunctionUtil {
  public static float[] fitPolynomial(float[] data) {
    int numCoeff = 6;
    int iterations = numCoeff * 4;
    float[] initial = new float[numCoeff];

    float[] initialStep = new float[numCoeff];
    Arrays.fill(initialStep, 1.0f);
    return optimize(1.0 / 512.0, iterations, initial, initialStep,
        new ImmutablePair<Float, Float>(-10.0f, 10.0f), data);
  }

  public static float evaluatePolynomial(float[] coeff, float[] data, boolean overwrite) {
    int len = data.length;
    float err = 0;
    long count = 0;
    for (int i = 0; i < len; i++) {
      float t = (float) i / (len - 1);
      float tinv = 1 - t;
      float diracStart = (i == 0) ? coeff[0] : 0;
      float diracEnd = (i == len - 1) ? coeff[1] : 0;
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
        data[i] = (float) eval;
      }
    }
    return err / count;
  }

  // CyclicCoordinateDescent
  public static float[] optimize(double tolerance, int iterations,
                                 float[] initial, float[] initialStep,
                                 Pair<Float, Float> bounds, float[] data) {
    float[] best = initial;
    float bestF = evaluatePolynomial(best, data, false);
    int maxDim = initial.length;
    for (int i = 0; i < iterations; ++i) {
      for (int dim = 0; dim < maxDim; ++dim) {
        float step = initialStep[dim];
        while (step > tolerance) {
          float[] left = best.clone();
          left[dim] = Math.max(bounds.getLeft(), best[dim] - step);
          float leftF = evaluatePolynomial(left, data, false);
          float[] right = best.clone();
          right[dim] = Math.min(bounds.getRight(), best[dim] + step);
          float rightF = evaluatePolynomial(right, data, false);
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

  public static float[] toFloat(List<Double> list) {
    float[] result = new float[list.size()];
    for (int i = 0; i < result.length; i++) {
      result[i] = list.get(i).floatValue();
    }
    return result;
  }

  /*
 * @param  tolerance if fitted array's deviation from weights is less than tolerance
 *         use the fitted, otherwise keep original weights.
 * @param  weights the curve you want to smooth
 * @return double errAndCoeff in the weights
   */
  public static double smooth(double tolerance, boolean toleranceIsPercentage, float[] weights) {
    // TODO use apache math's PolynomialCurveFitter
    float[] best = FunctionUtil.fitPolynomial(weights);
    double errAndCoeff = FunctionUtil.evaluatePolynomial(best, weights, false);
    if (toleranceIsPercentage) {
      double absMean = getAbsMean(weights);
      return smoothInternal(errAndCoeff, tolerance * absMean, best, weights) / absMean;
    } else {
      return smoothInternal(errAndCoeff, tolerance, best, weights);
    }
  }

  private static double smoothInternal(
      double errAndCoeff, double tolerance, float[] best, float[] weights) {
    if (errAndCoeff < tolerance) {
      FunctionUtil.evaluatePolynomial(best, weights, true);
    }
    return errAndCoeff;
  }

  public static double getAbsMean(float[] weights) {
    double sum = 0;
    for (float f : weights) {
      sum += Math.abs(f);
    }
    return  sum / weights.length;
  }
}
