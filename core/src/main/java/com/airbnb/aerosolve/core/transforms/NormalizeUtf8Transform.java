package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.transforms.types.StringTransform;

import java.text.Normalizer;

import com.typesafe.config.Config;

/**
 * Normalizes strings to UTF-8 NFC, NFD, NFKC or NFKD form (NFD by default)
 * "field1" specifies the key of the feature
 * "normalization_form" optionally specifies whether to use NFC, NFD, NFKC or NFKD form
 * "output" optionally specifies the key of the output feature, if it is not given the transform
 * overwrites / replaces the input feature
 */
public class NormalizeUtf8Transform extends StringTransform {
  public static final Normalizer.Form DEFAULT_NORMALIZATION_FORM = Normalizer.Form.NFD;

  private Normalizer.Form normalizationForm;

  @Override
  public void init(Config config, String key) {
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
  }

  @Override
  public String processString(String rawString) {
    if (rawString == null) {
      return null;
    }

    return Normalizer.normalize(rawString, normalizationForm);
  }
}
