package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.models.AbstractModel;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.types.ConfigurableTransform;
import com.typesafe.config.Config;
import java.util.List;
import java.util.function.Function;
import lombok.AccessLevel;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

/**
 * Created by hector_yee on 8/25/14.
 * A transform that accepts a list of other transforms and applies them as a group
 * in the order specified by the list.
 */
@Slf4j
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

  private Function<MultiFamilyVector, MultiFamilyVector> bigTransform;

  @Override
  public ListTransform configure(Config config, String key) {
    List<String> transformKeys = config.getStringList(key + ".transforms");
    for (String transformKey : transformKeys) {
      Transform<MultiFamilyVector> tmpTransform =
          TransformFactory.createTransform(config, transformKey, registry, model);
      if (bigTransform == null) {
        bigTransform = tmpTransform;
      } else {
        bigTransform = bigTransform.andThen(tmpTransform);
      }
    }
    return this;
  }

  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    bigTransform.apply(featureVector);
  }
}
