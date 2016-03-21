package com.airbnb.aerosolve.core.util;

import lombok.experimental.Builder;
import lombok.extern.slf4j.Slf4j;

/*
  weibull(x) = exp(a*(x)^k + b)
  default max x is Double.MAX_VALUE
 */
@Slf4j @Builder public class Weibull {
  final private double k, a, b, maxX;

  public WeibullBuilder defaultBuilder() {
    return Weibull.builder().maxX(Double.MAX_VALUE);
  }

  public double getWeibull(double x) {
    return Math.exp(a * Math.pow(x, k) + b);
  }

  /*
     weibull(x) = exp(a*(x)^k + b)
     given value of weibull(x), return x
   */
  public double getXWithOutput(double output) {
    // Below are the logic to compute the x:
    //    val t = (Math.log(p) - b) / a
    //    if (t > 0) ratio = math.min(math.pow(t, 1.0 / k), 2)
    //    else x = 0.0
    if (Math.abs(a) < 1e-10) {
      if (b > Math.log(output)) {
        return maxX;
      } else {
        return 0.0;
      }
    }

    // a is not too small.
    double t = (Math.log(output) - b) / a;
    log.debug("k = {}, a = {}, b = {}, t = {}", k, a, b, t);
    if (t > 0) {
      return Math.min(Math.pow(t, 1.0/k), maxX);
    } else {
      return 0.0;
    }
  }
}
