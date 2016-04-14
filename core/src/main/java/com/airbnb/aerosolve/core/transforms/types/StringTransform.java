package com.airbnb.aerosolve.core.transforms.types;

import com.airbnb.aerosolve.core.perf.FeatureValue;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import java.util.HashSet;

/**
 * Abstract representation of a transform that processes all strings in a string feature and
 * outputs a new string feature or overwrites /replaces the input string feature.
 * "field1" specifies the key of the feature
 * "output" optionally specifies the key of the output feature, if it is not given the transform
 * overwrites / replaces the input feature
 */
public abstract class StringTransform<T extends StringTransform> extends BaseFeaturesTransform<T> {
  @Override
  public void doTransform(MultiFamilyVector featureVector) {
    HashSet<String> processedStrings = new HashSet<>();

    for (FeatureValue featureValue : getInput(featureVector)) {
      if (featureValue.feature().name() != null) {
        processedStrings.add(processString(featureValue.feature().name()));
      }
    }

    // Check reference equality to determine whether the output should overwrite the input
    if (outputFamily == inputFamily) {
      // TODO (Brad): Not sure how I feel about doing this.  Are we sure?
      featureVector.remove(inputFamily);
    }

    for (String string : processedStrings) {
      featureVector.putString(outputFamily.feature(string));
    }
  }

  public abstract String processString(String rawString);
}
