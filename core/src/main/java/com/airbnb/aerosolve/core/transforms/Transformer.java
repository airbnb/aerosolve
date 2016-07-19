package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

import com.typesafe.config.Config;

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
      items.forEach(this::transformItem);
    }
  }

  /**
   * Apply combined transform to a (already context-combined) feature vector
   */
  public void transformCombined(FeatureVector combined) {
    if (combinedTransform != null && combined != null) {
      combinedTransform.doTransform(combined);
    }
  }

  /**
   * Apply combined transform to a stream of (already context-combined) feature vector
   */
  public void transformCombined(Iterable<FeatureVector> combined) {
    if (combinedTransform != null && combined != null) {
      combinedTransform.doTransform(combined);
    }
  }

  /**
   * In place apply all the transforms to the context and items,
   * add context to examples,
   * and apply the combined transform to now combined examples.
   */
  public void combineContextAndItems(Example examples) {
    transformContext(examples.context);
    transformItems(examples.example);
    addContextToItemsAndTransform(examples);
  }

  /**
   * Adds the context to items and applies the combined transform
   */
  public void addContextToItemsAndTransform(Example examples) {
    addContextToItems(examples);
    transformCombined(examples.example);
  }

  /**
   * Adds the context's features to examples' features
   */
  public void addContextToItems(Example examples) {
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
      addContextToItem(contextStringFeatures, contextFloatFeatures, contextDenseFeatures, item);
    }
  }

  /**
   * Adds context features to an individual feature vector
   */
  private void addContextToItem(Map<String, Set<String>> contextStringFeatures,
                                Map<String, Map<String, Double>> contextFloatFeatures,
                                Map<String, List<Double>> contextDenseFeatures,
                                FeatureVector item) {
    if (contextStringFeatures != null) {
      if (item.getStringFeatures() == null) {
        item.setStringFeatures(new HashMap<>());
      }
      Map<String, Set<String>> itemStringFeatures = item.getStringFeatures();
      for (Map.Entry<String, Set<String>> stringFeature : contextStringFeatures.entrySet()) {
        Set<String> stringFeatureValueCopy = new HashSet<>(stringFeature.getValue());
        itemStringFeatures.put(stringFeature.getKey(), stringFeatureValueCopy);
      }
    }
    if (contextFloatFeatures != null) {
      if (item.getFloatFeatures() == null) {
        item.setFloatFeatures(new HashMap<>());
      }
      Map<String, Map<String, Double>> itemFloatFeatures = item.getFloatFeatures();
      for (Map.Entry<String, Map<String, Double>> floatFeature : contextFloatFeatures.entrySet()) {
        Map<String, Double> floatFeatureValueCopy = new HashMap<>(floatFeature.getValue());
        itemFloatFeatures.put(floatFeature.getKey(), floatFeatureValueCopy);
      }
    }
    if (contextDenseFeatures != null) {
      if (item.getDenseFeatures() == null) {
        item.setDenseFeatures(new HashMap<>());
      }
      Map<String, List<Double>> itemDenseFeatures = item.getDenseFeatures();
      for (Map.Entry<String, List<Double>> denseFeature : contextDenseFeatures.entrySet()) {
        List<Double> denseFeatureValueCopy = new ArrayList<>(denseFeature.getValue());
        itemDenseFeatures.put(denseFeature.getKey(), denseFeatureValueCopy);
      }
    }
  }
}
