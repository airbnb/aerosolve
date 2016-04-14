package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.models.AbstractModel;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.ConfigurableTransform;
import com.airbnb.aerosolve.core.util.Util;
import com.typesafe.config.Config;
import java.util.Map;
import lombok.Synchronized;

/**
 * Created by hector_yee on 8/25/14.
 */
public class TransformFactory {

  private static Map<String, Class<? extends ConfigurableTransform>> TRANSFORM_CLASSES;

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
    if (TRANSFORM_CLASSES == null) {
      loadTransformMap();
    }

    Class<? extends ConfigurableTransform> clazz = TRANSFORM_CLASSES.get(transformName);
    if (clazz == null) {
      throw new IllegalArgumentException(
          String.format("No transform exists with name %s", transformName));
    }
    try {
      ConfigurableTransform<?> transform = clazz.newInstance();
      // (Brad): It's kind of awkward we have to do all this initialization. I tried to make
      // Transforms immutable and was hoping to use Builder inheritance to make them buildable in
      // a logical way.  But Builder inheritance is a mess in Java and constructor inheritance
      // leads to a bunch of hard to understand boilerplate in the concrete classes. So, this is
      // what we have.  Every Transform has to have a package private constructor. They can't be
      // initialized any other way.  So as long as we do all the steps correctly and in the right
      // order here, we should be safe.
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

  @Synchronized
  private static void loadTransformMap() {
    if (TRANSFORM_CLASSES != null) {
      return;
    }
    TRANSFORM_CLASSES = Util.loadFactoryNamesFromPackage(ConfigurableTransform.class,
                                                         "com.airbnb.aerosolve.core.transforms",
                                                         "Transform");
  }
}
