package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.FeatureVector;

/**
 * Created by hector_yee on 8/25/14.
 * Base class for models
 */
interface Model {
  // Scores a single item. The transforms should already have been applied to
  // the context and item and combined item.
  float scoreItem(FeatureVector combinedItem);
  // Debug scores a single item. These are explanations for why a model
  // came up with the score.
  float debugScoreItem(FeatureVector combinedItem,
                       StringBuilder builder);
}
