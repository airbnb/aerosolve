package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.models.AbstractModel;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.typesafe.config.Config;
import java.io.Serializable;
import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * Created by hector_yee on 8/25/14.
 * Base class for feature transforms.
 */
@FunctionalInterface
public interface Transform<T extends FeatureVector> extends Function<T, T> {}
