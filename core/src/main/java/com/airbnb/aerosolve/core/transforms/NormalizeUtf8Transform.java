package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.util.Util;

import java.text.Normalizer;
import java.util.Map;
import java.util.Set;

import com.typesafe.config.Config;

/**
 * Normalizes strings to UTF-8 NFC, NFD, NFKC or NFKD form (NFD by default)
 * "field1" specifies the key of the feature
 * "normalization_form" optionally specifies whether to use NFC, NFD, NFKC or NFKD form
 */
public class NormalizeUtf8Transform extends Transform {
  public static final Normalizer.Form DEFAULT_NORMALIZATION_FORM = Normalizer.Form.NFD;

  private String fieldName1;
  private Normalizer.Form normalizationForm;
  private String outputName;

  @Override
  public void configure(Config config, String key) {
    fieldName1 = config.getString(key + ".field1");
    String normalizationFormString = DEFAULT_NORMALIZATION_FORM.name();
    if (config.hasPath(key + ".normalization_form")) {
      normalizationFormString = config.getString(key + ".normalization_form");
    }
    if (normalizationFormString.equalsIgnoreCase("NFC")) {
      normalizationForm = Normalizer.Form.NFC;
    } else if (normalizationFormString.equalsIgnoreCase("NFD")) {
      normalizationForm = Normalizer.Form.NFD;
    } else if (normalizationFormString.equalsIgnoreCase("NFKC")) {
      normalizationForm = Normalizer.Form.NFKC;
    } else if (normalizationFormString.equalsIgnoreCase("NFKD")) {
      normalizationForm = Normalizer.Form.NFKD;
    } else {
      normalizationForm = DEFAULT_NORMALIZATION_FORM;
    }
    outputName = config.getString(key + ".output");
  }

  @Override
  public void doTransform(FeatureVector featureVector) {
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    if (stringFeatures == null) {
      return;
    }

    Set<String> feature1 = stringFeatures.get(fieldName1);
    if (feature1 == null) {
      return;
    }

    Set<String> output = Util.getOrCreateStringFeature(outputName, stringFeatures);

    for (String rawString : feature1) {
      if (rawString == null) continue;
      String normalizedString = Normalizer.normalize(rawString, normalizationForm);
      output.add(normalizedString);
    }
  }
}
