package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.models.AbstractModel;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.typesafe.config.Config;
import java.io.Serializable;

public class Transformer implements Serializable {

  private static final long serialVersionUID = 1569952057032186608L;
  // The transforms to be applied to the context, item and combined
  // (context | item) respectively.
  private final Transform<MultiFamilyVector> contextTransform;
  private final Transform<MultiFamilyVector> itemTransform;
  private final Transform<MultiFamilyVector> combinedTransform;

  public Transformer(Config config, String key, FeatureRegistry registry) {
    this(config, key, registry, null);
  }

  public Transformer(Config config, String key, FeatureRegistry registry, AbstractModel model) {
    // Configures the model transforms.
    // context_transform : name of ListTransform to apply to context
    // item_transform : name of ListTransform to apply to each item
    // combined_transform : name of ListTransform to apply to each (item context) pair
    String contextTransformName = config.getString(key + ".context_transform");
    contextTransform = TransformFactory.createTransform(config, contextTransformName,
                                                        registry, model);
    String itemTransformName = config.getString(key + ".item_transform");
    itemTransform = TransformFactory.createTransform(config, itemTransformName,
                                                     registry, model);
    String combinedTransformName = config.getString(key + ".combined_transform");
    combinedTransform = TransformFactory.createTransform(config, combinedTransformName,
                                                         registry, model);
  }

  public Transform<MultiFamilyVector> getContextTransform() {
    return contextTransform;
  }

  public Transform<MultiFamilyVector> getItemTransform() {
    return itemTransform;
  }

  public Transform<MultiFamilyVector> getCombinedTransform() {
    return combinedTransform;
  }
}
