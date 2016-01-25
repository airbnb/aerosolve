package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;

import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

public class Transformer implements Serializable {

  private static final long serialVersionUID = 1569952057032186608L;
  // The transforms to be applied to the context, item and combined
  // (context | item) respectively.
  private final Transform contextTransform;
  private final Transform itemTransform;
  private final Transform combinedTransform;

  public Transformer(Config config, String key) {
    // Configures the model transforms.
    // context_transform : name of ListTransform to apply to context
    // item_transform : name of ListTransform to apply to each item
    // combined_transform : name of ListTransform to apply to each (item context) pair
    String contextTransformName = config.getString(key + ".context_transform");
    contextTransform = TransformFactory.createTransform(config, contextTransformName);
    String itemTransformName = config.getString(key + ".item_transform");
    itemTransform = TransformFactory.createTransform(config, itemTransformName);
    String combinedTransformName = config.getString(key + ".combined_transform");
    combinedTransform = TransformFactory.createTransform(config, combinedTransformName);
  }

  // Helper functions for transforming context, items or combined feature vectors.
  public void transformContext(FeatureVector context) {
    if (contextTransform != null && context != null) {
      contextTransform.doTransform(context);
    }
  }

  public void transformItem(FeatureVector item) {
    if (itemTransform != null && item != null) {
      itemTransform.doTransform(item);
    }
  }

  public void transformItems(List<FeatureVector> items) {
    if (items != null) {
      for (FeatureVector item : items) {
        transformItem(item);
      }
    }
  }

  public void transformCombined(FeatureVector combined) {
    if (combinedTransform != null && combined != null) {
      combinedTransform.doTransform(combined);
    }
  }

  // In place apply all the transforms to the context and items
  // and apply the combined transform to items.
  public void combineContextAndItems(Example examples) {
    transformContext(examples.context);
    transformItems(examples.example);
    addContextToItemsAndTransform(examples);
  }

  // Adds the context to items and applies the combined transform
  public void addContextToItemsAndTransform(Example examples) {
    Map<String, Set<String>> contextStringFeatures = null;
    Map<String, Map<String, Double>> contextFloatFeatures = null;
    Map<String, List<Double>> contextDenseFeatures = null;
    if (examples.context != null) {
      if (examples.context.stringFeatures != null) {
        contextStringFeatures = examples.context.getStringFeatures();
      }
      if (examples.context.floatFeatures != null) {
        contextFloatFeatures = examples.context.getFloatFeatures();
      }
      if (examples.context.denseFeatures != null) {
        contextDenseFeatures = examples.context.getDenseFeatures();
      }
    }
    for (FeatureVector item : examples.example) {
      addContextToItemAndTransform(
          contextStringFeatures, contextFloatFeatures, contextDenseFeatures, item);
    }
  }

  public void addContextToItemAndTransform(Map<String, Set<String>> contextStringFeatures,
                                           Map<String, Map<String, Double>> contextFloatFeatures,
                                           Map<String, List<Double>> contextDenseFeatures,
                                           FeatureVector item) {
    Map<String, Set<String>> itemStringFeatures = item.getStringFeatures();
    Map<String, Map<String, Double>> itemFloatFeatures = item.getFloatFeatures();
    Map<String, List<Double>> itemDenseFeatures = item.getDenseFeatures();
    if (item.getStringFeatures() == null) {
      item.setStringFeatures(contextStringFeatures);
    } else if (contextStringFeatures != null) {
      itemStringFeatures.putAll(contextStringFeatures);
    }
    if (item.getFloatFeatures() == null) {
      item.setFloatFeatures(contextFloatFeatures);
    } else if (contextFloatFeatures != null) {
      itemFloatFeatures.putAll(contextFloatFeatures);
    }
    if (item.getDenseFeatures() == null) {
      item.setDenseFeatures(contextDenseFeatures);
    } else if (contextDenseFeatures != null) {
      itemDenseFeatures.putAll(contextDenseFeatures);
    }
    transformCombined(item);
  }
}
