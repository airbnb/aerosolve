package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import java.io.Serializable;
import java.util.function.Function;

/**
 * Created by hector_yee on 8/25/14.
 * Base class for feature transforms.
 */
@FunctionalInterface
public interface Transform<T extends FeatureVector> extends Function<T, T>, Serializable {}
