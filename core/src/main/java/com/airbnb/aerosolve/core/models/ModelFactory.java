package com.airbnb.aerosolve.core.models;

import java.io.BufferedReader;
import java.io.IOException;

import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.util.Util;
import com.google.common.base.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ModelFactory {
  private static final Logger log = LoggerFactory.getLogger(ModelFactory.class);

  private ModelFactory() {
  }
  // Creates
  public static AbstractModel createByName(String name) {
    switch (name) {
      case "linear": return new LinearModel();
      case "maxout": return new MaxoutModel();
      case "spline": return new SplineModel();
      case "boosted_stumps": return new BoostedStumpsModel();
      case "decision_tree": return new DecisionTreeModel();
      case "forest": return new ForestModel();
    }
    log.error("Could not create model of type " + name);
    return null;
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
