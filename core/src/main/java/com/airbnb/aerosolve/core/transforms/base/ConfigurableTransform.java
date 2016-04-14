package com.airbnb.aerosolve.core.transforms.base;

import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.Transform;
import com.google.common.collect.ImmutableSet;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigObject;
import com.typesafe.config.ConfigValue;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Collectors;
import lombok.Getter;

import javax.validation.ConstraintViolation;
import javax.validation.Validation;
import javax.validation.ValidatorFactory;
import javax.validation.constraints.NotNull;

/**
 *
 */
@SuppressWarnings("unchecked")
public abstract class ConfigurableTransform<T extends ConfigurableTransform>
    implements Transform<MultiFamilyVector> {
  private static final ValidatorFactory
      VALIDATION_FACTORY = Validation.buildDefaultValidatorFactory();

  private boolean setupComplete = false;

  @Getter
  @NotNull
  protected FeatureRegistry registry;

  public T registry(FeatureRegistry registry) {
    this.registry = registry;
    return (T) this;
  }

  protected ConfigurableTransform() {
  }

  public abstract T configure(Config config, String key);

  abstract protected void doTransform(MultiFamilyVector vector);

  @Override
  public MultiFamilyVector apply(MultiFamilyVector vector) {
    if (!setupComplete) {
      validate();
      setup();
      setupComplete = true;
    }
    if (checkPreconditions(vector)) {
      doTransform(vector);
    }
    return vector;
  }

  protected boolean checkPreconditions(MultiFamilyVector vector) {
    return true;
  }

  protected void setup() {
    // Do nothing. Override.
  }

  protected void validate() {
    // This sort of sucks and I wish I could make it immutable instead.
    // This is a small speed hack. If the user mutates the Transformer after this call,
    // things can get bad.  But we don't have immutability right now and I don't want to
    // much things up with a dirty flag.  So this will have to do for the time being.
    if (setupComplete) {
      return;
    }
    Set<ConstraintViolation<ConfigurableTransform<?>>> violations =
        VALIDATION_FACTORY.getValidator().validate(this);
    int numViolations = violations.size();

    if (numViolations > 0) {
      String violationMessage = String.join("\n", violations.stream()
          .map(v -> v.getPropertyPath() + " " + v.getMessage()).collect(Collectors.toList()));
      throw new IllegalArgumentException(
          String.format("Transformer failed validation with %d violations:\n %s", numViolations,
                        violationMessage));
    }
  }

  protected static String stringFromConfig(Config config, String key, String field) {
    return stringFromConfig(config, key, field, true);
  }

  protected static String stringFromConfig(Config config, String key, String field,
                                           boolean failIfAbsent) {
    String path = key + field;
    if (failIfAbsent || config.hasPath(path)) {
      // This fails with an exception if the path doesn't exist.
      return config.getString(key + field);
    }
    return null;
  }

  protected static Double doubleFromConfig(Config config, String key, String field) {
    return doubleFromConfig(config, key, field, true, null);
  }

  protected static Double doubleFromConfig(Config config, String key, String field,
                                           boolean failIfAbsent) {
    return doubleFromConfig(config, key, field, failIfAbsent, null);
  }

  protected static Double doubleFromConfig(Config config, String key, String field,
                                           boolean failIfAbsent, Double defaultValue) {
    String path = key + field;
    if (failIfAbsent || config.hasPath(path)) {
      return config.getDouble(path);
    }
    return defaultValue;
  }

  protected static Integer intFromConfig(Config config, String key, String field) {
    return intFromConfig(config, key, field, true, null);
  }

  protected static Integer intFromConfig(Config config, String key, String field,
                                         boolean failIfAbsent) {
    return intFromConfig(config, key, field, failIfAbsent, null);
  }

  protected static Integer intFromConfig(Config config, String key, String field,
                                         boolean failIfAbsent,
                                         Integer defaultValue) {
    String fullPath = key + field;
    if (failIfAbsent || config.hasPath(fullPath)) {
      return config.getInt(fullPath);
    }
    return defaultValue;
  }

  protected static boolean booleanFromConfig(Config config, String key, String field) {
    String fullPath = key + field;
    return config.hasPath(fullPath) && config.getBoolean(fullPath);
  }

  protected static Set<String> stringSetFromConfig(Config config, String key, String field,
                                                   boolean failIfAbsent) {
    List<String> result = stringListFromConfig(config, key, field, failIfAbsent);
    return result == null ? null : ImmutableSet.copyOf(result);
  }

  protected static List<String> stringListFromConfig(Config config, String key, String field,
                                                     boolean failIfAbsent) {
    String fullKey = key + field;
    if (failIfAbsent || config.hasPath(fullKey)) {
      return config.getStringList(fullKey);
    }
    return null;
  }

  protected static List<Double> doubleListFromConfig(Config config, String key, String field,
                                                     boolean failIfAbsent) {
    String fullKey = key + field;
    if (failIfAbsent || config.hasPath(fullKey)) {
      return config.getDoubleList(fullKey);
    }
    return null;
  }

  protected static TreeMap<Double, Double> doubleTreeMapFromConfig(Config config, String key,
                                                                   String field,
                                                                   boolean failIfAbsent) {
    String fullPath = key + field;
    if (failIfAbsent || config.hasPath(fullPath)) {
      TreeMap<Double, Double> parsedTokensMap = new TreeMap<>();
      for (ConfigObject configObject : config.getObjectList(fullPath)) {
        List<Map.Entry<String, ConfigValue>> entries = new ArrayList<>(configObject.entrySet());
        parsedTokensMap.put(Double.parseDouble(entries.get(0).getKey()),
                            Double.parseDouble(entries.get(0).getValue().unwrapped().toString()));
      }

      return parsedTokensMap;
    }
    return null;
  }

  protected static Map<String, String> stringMapFromConfig(Config config, String key, String field,
                                                           boolean failIfAbsent) {
    String fullPath = key + field;
    if (failIfAbsent || config.hasPath(fullPath)) {
      return config.getObjectList(fullPath)
          .stream()
          .flatMap((ConfigObject o) -> o.unwrapped().entrySet().stream())
          .collect(Collectors.toMap(Map.Entry::getKey, e -> (String) e.getValue()));
    }
    return null;
  }


  protected static String getTransformType(Config config, String key) {
    return stringFromConfig(config, key, ".transform", false);
  }
}
