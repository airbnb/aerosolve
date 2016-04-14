package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.transforms.base.BaseFeaturesTransform;
import com.typesafe.config.Config;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;

/**
 * Tokenizes and counts strings using a regex and optionally generates bigrams from the tokens
 * "field1" specifies the key of the feature
 * "regex" specifies the regex used to tokenize
 * "generateBigrams" specifies whether bigrams should also be generated
 */
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class DefaultStringTokenizerTransform
    extends BaseFeaturesTransform<DefaultStringTokenizerTransform> {
  public static final String BIGRAM_SEPARATOR = " ";

  // TODO (Brad): Is there a good default regex?
  private String regex = " ";
  private boolean generateBigrams = false;
  private String bigramsOutputFamilyName;


  @Setter(AccessLevel.NONE)
  private Family bigramsOutputFamily;

  @Override
  public DefaultStringTokenizerTransform configure(Config config, String key) {
    return super.configure(config, key)
        .regex(stringFromConfig(config, key, ".regex"))
        .generateBigrams(booleanFromConfig(config, key, ".generate_bigrams"))
        .bigramsOutputFamilyName(stringFromConfig(config, key, ".bigrams_output"));
  }

  @Override
  protected void setup() {
    super.setup();
    if (generateBigrams && bigramsOutputFamilyName != null) {
      bigramsOutputFamily = registry.family(bigramsOutputFamilyName);
    }
  }

  @Override
  protected void doTransform(MultiFamilyVector featureVector) {

    for (FeatureValue value : getInput(featureVector)) {
      if (value.feature().name() == null) {
        continue;
      }

      String previousToken = null;
      for (String token : value.feature().name().split(regex)) {
        if (token.length() == 0) continue;
        incrementOutput(outputFamily.feature(token), featureVector);
        if (generateBigrams) {
          if (previousToken != null) {
            String bigram = previousToken + BIGRAM_SEPARATOR + token;
            incrementOutput(bigramsOutputFamily.feature(bigram), featureVector);
          }
          previousToken = token;
        }
      }
    }
  }

  private static void incrementOutput(Feature feature, FeatureVector vector) {
    if (vector.containsKey(feature)) {
      double count = vector.get(feature);
      vector.put(feature, (count + 1.0));
    } else {
      vector.put(feature, 1.0);
    }
  }
}
