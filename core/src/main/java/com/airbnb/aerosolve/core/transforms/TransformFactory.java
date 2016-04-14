package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.models.AbstractModel;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.types.ConfigurableTransform;
import com.google.common.base.CaseFormat;
import com.typesafe.config.Config;
import java.lang.annotation.Annotation;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.commons.lang3.tuple.Pair;
import org.reflections.Reflections;

/**
 * Created by hector_yee on 8/25/14.
 */
public class TransformFactory {

  private static Map<String, Class<? extends ConfigurableTransform>> TRANSFORMS;

  public static Transform<MultiFamilyVector> createTransform(Config config, String key,
                                                             FeatureRegistry registry,
                                                             AbstractModel model) {
    if (config == null || key == null) {
      return null;
    }
    String transformName = config.getString(key + ".transform");
    if (transformName == null) {
      return null;
    }

    // TODO (Brad): I don't love that this is a static initialization.  Can lead to awkward bugs but
    // it's probably tricky to make clients use a singleton instance of this class without Guice.
    // Not sure a static singleton in the class is any better.
    if (TRANSFORMS == null) {
      loadTransformMap();
    }

    Class<? extends ConfigurableTransform> clazz = TRANSFORMS.get(transformName);
    if (clazz == null) {
      throw new IllegalArgumentException(
          String.format("No transform exists with name %s", transformName));
    }
    try {
      ConfigurableTransform<?> transform = clazz.newInstance();
      transform.registry(registry);
      if (transform instanceof ModelAware && model != null) {
        ((ModelAware)transform).model(model);
      }
      transform.configure(config, key);
      return transform;
    } catch (InstantiationException | IllegalAccessException e) {
      throw new IllegalStateException(
          String.format("There was an error instantiating Transform of class %s", clazz.getName()));
    }
  }

  private static synchronized void loadTransformMap() {
    if (TRANSFORMS != null) {
      return;
    }
    Reflections reflections = new Reflections("com.airbnb.aerosolve.core.transforms");
    TRANSFORMS = reflections.getSubTypesOf(ConfigurableTransform.class).stream()
        .filter(clazz -> !clazz.isInterface() && !Modifier.isAbstract(clazz.getModifiers()))
        .flatMap(clazz -> getClassNames(clazz).stream())
        .collect(Collectors.toMap(Pair::getKey, Pair::getValue));
  }

  private static List<Pair<String, Class<? extends ConfigurableTransform>>> getClassNames(
      Class<? extends ConfigurableTransform> clazz) {
    List<Pair<String, Class<? extends ConfigurableTransform>>> result = new ArrayList<>();

    String baseName = clazz.getSimpleName();

    // The default name is lower case snake case without Transform at the end.
    if (baseName.endsWith("Transform")) {
      baseName = baseName.substring(0, baseName.length() - 9);
    }
    baseName = CaseFormat.UPPER_CAMEL.to(CaseFormat.LOWER_UNDERSCORE, baseName);
    result.add(Pair.of(baseName, clazz));

    // Handle any old names we used to use that are annotated on the class.
    if (clazz.isAnnotationPresent(LegacyNames.class)) {
      LegacyNames legacyNames = clazz.getAnnotation(LegacyNames.class);
      for (String legacyName : legacyNames.value()) {
        result.add(Pair.of(legacyName, clazz));
      }
    }
    return result;
  }
}
