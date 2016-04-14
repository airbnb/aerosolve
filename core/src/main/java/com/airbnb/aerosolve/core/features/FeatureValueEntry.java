package com.airbnb.aerosolve.core.features;

import it.unimi.dsi.fastutil.objects.Object2DoubleMap;

/**
 *
 */
public interface FeatureValueEntry extends Object2DoubleMap.Entry<Feature>, FeatureValue {
  void feature(Feature feature);
}
