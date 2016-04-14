package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.util.Util;
import com.google.common.base.Optional;
import java.io.BufferedReader;
import java.io.IOException;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public final class ModelFactory {

  private ModelFactory() {
  }

  // Creates
  @SuppressWarnings("deprecation")
  public static AbstractModel createByName(String name, FeatureRegistry registry) {
    switch (name) {
      case "linear": return new LinearModel(registry);
      case "maxout": return new MaxoutModel(registry);
      case "spline": return new SplineModel(registry);
      case "boosted_stumps": return new BoostedStumpsModel(registry);
      case "decision_tree": return new DecisionTreeModel(registry);
      case "forest": return new ForestModel(registry);
      case "additive": return new AdditiveModel(registry);
      case "kernel" : return new KernelModel(registry);
      case "full_rank_linear" : return new FullRankLinearModel(registry);
      case "low_rank_linear" : return new LowRankLinearModel(registry);
      case "multilayer_perceptron" : return new MlpModel(registry);
    }
    log.error("Could not create model of type " + name);
    return null;
  }

  public static Optional<AbstractModel> createFromReader(BufferedReader reader,
                                                         FeatureRegistry registry) throws IOException {
    Optional<AbstractModel> model = Optional.absent();
    String headerLine = reader.readLine();
    ModelRecord record = Util.decodeModel(headerLine);
    if (record == null) {
      log.error("Could not decode header " + headerLine);
      return model;
    }
    ModelHeader header = record.getModelHeader();
    if (header != null) {
      AbstractModel result = createByName(header.getModelType(), registry);
      if (result != null) {
        result.loadInternal(header, reader);
        model = Optional.of(result);
      }
    }
    return model;
  }
}
