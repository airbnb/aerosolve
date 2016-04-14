package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.models.AbstractModel;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.ConfigurableTransform;
import com.typesafe.config.Config;
import java.util.List;
import java.util.function.Function;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * Created by hector_yee on 8/25/14.
 * A transform that accepts a list of other transforms and applies them as a group
 * in the order specified by the list.
 */
@Slf4j
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class ListTransform extends ConfigurableTransform<ListTransform>
    implements ModelAware<ListTransform> {

  private AbstractModel model;

  @Override
  public AbstractModel model() {
    return model;
  }

  @Override
  public ListTransform model(AbstractModel model) {
    this.model = model;
    return this;
  }

  // Starts with identity to make null cases easier.
  private Transform<MultiFamilyVector> bigTransform = f -> f;

  @Override
  public ListTransform configure(Config config, String key) {
    List<String> transformKeys = config.getStringList(key + ".transforms");
    for (String transformKey : transformKeys) {
      Transform<MultiFamilyVector> tmpTransform =
          TransformFactory.createTransform(config, transformKey, registry, model);
      bigTransform = (Transform) bigTransform.andThen(tmpTransform);
    }
    return this;
  }

  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    bigTransform.apply(featureVector);
  }
}
