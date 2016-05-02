package com.airbnb.aerosolve.core.util;

/**
 * Utility class for methods that are commonly used by transforms and should be shared.
 * TODO: go through all transforms and add all common methods here.
 * TODO: go through Util class and add all transform related / exclusive methods here.
 */
public class TransformUtil {
    public static Double quantize(double val, double delta) {
      Double mult = val / delta;
      return delta * mult.intValue();
    }
}
