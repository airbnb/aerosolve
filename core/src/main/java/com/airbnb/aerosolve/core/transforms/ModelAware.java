package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.models.AbstractModel;

/**
 *
 */
public interface ModelAware<T> {

  AbstractModel model();

  T model(AbstractModel model);
}
