package com.airbnb.aerosolve.core.models;

import com.airbnb.aerosolve.core.ModelHeader;
import com.airbnb.aerosolve.core.ModelRecord;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.util.Util;
import com.google.common.base.Optional;
import java.io.BufferedReader;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.HashMap;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public final class ModelFactory {

  private ModelFactory() {
  }

  /**
   * This a Map from the type name of a model ("linear" for instance) to a constructor that takes
   * a single argument of type FeatureRegistry. This caches most of the reflection and makes
   * instantiation easy.  But we don't have to explicitly include new models in this class. They
   * are found from the classpath.
   */
  private static Map<String, Constructor<? extends AbstractModel>> MODEL_CONSTRUCTORS;

  public static AbstractModel createByName(String name, FeatureRegistry registry) {
    if (MODEL_CONSTRUCTORS == null) {
      loadModelConstructors();
    }
    Constructor<? extends AbstractModel> constructor = MODEL_CONSTRUCTORS.get(name);
    if (constructor == null) {
      throw new IllegalArgumentException(
          String.format("No model exists with name %s", name));
    }
    try {
      return constructor.newInstance(registry);
    } catch (InstantiationException | IllegalAccessException | InvocationTargetException e) {
      throw new IllegalStateException(
          String.format("There was an error instantiating Model of type %s : %s",
                        name, e.getMessage()), e);
    }
  }

  private static synchronized void loadModelConstructors() {
    if (MODEL_CONSTRUCTORS != null) {
      return;
    }
    MODEL_CONSTRUCTORS = Util.loadConstructorsFromPackage(AbstractModel.class,
                                                          "com.airbnb.aerosolve.core.models",
                                                          "Model",
                                                          FeatureRegistry.class);
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
