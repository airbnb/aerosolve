package com.airbnb.aerosolve.core.transforms.base;

import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.typesafe.config.Config;
import it.unimi.dsi.fastutil.objects.Reference2ObjectMap;
import it.unimi.dsi.fastutil.objects.Reference2ObjectOpenHashMap;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Spliterator;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import lombok.Getter;

import javax.validation.constraints.NotNull;

@SuppressWarnings("unchecked")
public abstract class BaseFeaturesTransform<T extends BaseFeaturesTransform>
    extends ConfigurableTransform<T> {
  @Getter
  @NotNull
  protected String inputFamilyName;

  @Getter
  protected String outputFamilyName;

  @Getter
  protected Set<String> inputFeatureNames;

  @Getter
  protected Set<String> excludedFeatureNames;

  protected BaseFeaturesTransform() {
  }

  protected Family inputFamily;
  protected Family outputFamily;
  protected List<Feature> inputFeatures;
  protected Reference2ObjectMap<Feature, Feature> outputFeatures;
  protected Set<Feature> excludedFeatures;

  public T inputFamilyName(String name) {
    this.inputFamilyName = name;

    return (T) this;
  }

  public T outputFamilyName(String name) {
    this.outputFamilyName = name;
    return (T) this;
  }

  public T inputFeatureNames(Set<String> names) {
    inputFeatureNames = names;
    return (T) this;
  }

  public T excludedFeatureNames(Set<String> names) {
    this.excludedFeatureNames = names == null ? null : ImmutableSet.copyOf(names);
    return (T) this;
  }

  @Override
  protected void setup() {
    super.setup();
    this.inputFamily = registry.family(inputFamilyName);
    this.outputFamily = outputFamilyName == null
                        ? this.inputFamily
                        : registry.family(outputFamilyName);
    if (inputFeatureNames != null) {
      inputFeatures = new ArrayList<>();
      for (String featureName : inputFeatureNames) {
        Feature feature = inputFamily.feature(featureName);
        inputFeatures.add(feature);
        if (outputFamily != inputFamily) {
          if (outputFeatures == null) {
            outputFeatures = new Reference2ObjectOpenHashMap<>();
          }
          outputFeatures.put(feature, outputFamily.feature(produceOutputFeatureName(featureName)));
        }
      }
    }
    if (excludedFeatureNames != null) {
      excludedFeatures = new HashSet<>();
      for (String featureName : excludedFeatureNames) {
        excludedFeatures.add(inputFamily.feature(featureName));
      }
    }
  }

  protected String produceOutputFeatureName(String featureName) {
    return featureName;
  }

  protected Iterable<? extends FeatureValue> getInput(MultiFamilyVector featureVector) {
    Spliterator<FeatureValue> spliterator;
    if (inputFeatures != null) {
      spliterator = featureVector.iterateMatching(inputFeatures).spliterator();
    } else {
      spliterator = featureVector.get(inputFamily).spliterator();
    }
    Stream<FeatureValue> stream = StreamSupport.stream(spliterator, false);
    if (excludedFeatures != null) {
      stream = stream.filter(value -> !excludedFeatures.contains(value.feature()));
    }
    return stream::iterator;
  }

  protected Feature produceOutputFeature(Feature feature) {
    if (outputFeatures != null) {
      return outputFeatures.get(feature);
    }
    // TODO (Brad): Handle the case where outputFamily == inputFamily and we're not going to do
    // any sort of transform here.
    return outputFamily.feature(produceOutputFeatureName(feature.name()));
  }

  @Override
  protected boolean checkPreconditions(MultiFamilyVector vector) {
    return super.checkPreconditions(vector) && vector.contains(inputFamily);
  }

  @Override
  public T configure(Config config, String key) {
    // There are two ways we specified input feature names in configs.  No one should specify
    // both So we try each one.
    Set<String> inputFeatureNames = stringSetFromConfig(config, key, ".keys", false);
    Set<String> selectFeatureNames = stringSetFromConfig(config, key, ".select_features", false);
    inputFeatureNames = inputFeatureNames == null ? selectFeatureNames :
                        selectFeatureNames == null ? inputFeatureNames :
                                 Sets.union(inputFeatureNames, selectFeatureNames);
    return (T)
        inputFamilyName(stringFromConfig(config, key, ".field1"))
        .outputFamilyName(stringFromConfig(config, key, ".output", false))
        .inputFeatureNames(inputFeatureNames)
        .excludedFeatureNames(stringSetFromConfig(config, key, ".exclude_features", false));
  }
}
