package com.airbnb.aerosolve.core.util;

/**
 * Created by christhetree on 30/04/16.
 */
public class TransformUtil {
    public static Double quantize(double val, double delta) {
      Double mult = val / delta;
      return delta * mult.intValue();
    }
}
