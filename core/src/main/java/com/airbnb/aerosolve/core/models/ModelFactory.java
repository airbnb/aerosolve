package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.util.Util;
import com.google.common.base.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;

public final class ModelFactory {
  private static final Logger log = LoggerFactory.getLogger(ModelFactory.class);

  private ModelFactory() {
  }

  // Creates
  @SuppressWarnings("deprecation")
  public static AbstractModel createByName(String name) {
    switch (name) {
      case "linear": return new LinearModel();
      case "maxout": return new MaxoutModel();
      case "spline": return new SplineModel();
      case "boosted_stumps": return new BoostedStumpsModel();
      case "decision_tree": return new DecisionTreeModel();
      case "forest": return new ForestModel();
      case "additive": return new AdditiveModel();
      case "kernel" : return new KernelModel();
      case "full_rank_linear" : return new FullRankLinearModel();
      case "low_rank_linear" : return new LowRankLinearModel();
      case "multilayer_perceptron" : return new MlpModel();
      default:
        log.info("Attempting to initialize " + name);
        try {
          return (AbstractModel) Class.forName(name).newInstance();
        } catch (Exception e) {
          log.error("Unable to initialize model by class name of " + name);
          throw new RuntimeException(e);
        }
    }
  }
  public static Optional<AbstractModel> createFromReader(BufferedReader reader) throws IOException {
    Optional<AbstractModel> model = Optional.absent();
    String headerLine = reader.readLine();
    ModelRecord record = Util.decodeModel(headerLine);
    if (record == null) {
      log.error("Could not decode header " + headerLine);
      return model;
    }
    ModelHeader header = record.getModelHeader();
    if (header != null) {
      AbstractModel result = createByName(header.getModelType());
      if (result != null) {
        result.loadInternal(header, reader);
        model = Optional.of(result);
      }
    }
    return model;
  }
}
