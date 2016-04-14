package com.airbnb.aerosolve.core.perf;

import it.unimi.dsi.fastutil.objects.Reference2DoubleMap;

/**
 *
 */
public interface FeatureValueEntry extends Reference2DoubleMap.Entry<Feature>, FeatureValue {
  Feature setFeature(Feature feature);
}
