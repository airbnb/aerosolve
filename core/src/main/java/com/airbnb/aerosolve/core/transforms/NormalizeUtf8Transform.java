package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.transforms.base.StringTransform;
import com.typesafe.config.Config;
import java.text.Normalizer;
import lombok.AccessLevel;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.experimental.Accessors;

/**
 * Normalizes strings to UTF-8 NFC, NFD, NFKC or NFKD form (NFD by default)
 * "field1" specifies the key of the feature
 * "normalization_form" optionally specifies whether to use NFC, NFD, NFKC or NFKD form
 * "output" optionally specifies the key of the output feature, if it is not given the transform
 * overwrites / replaces the input feature
 */
// This is what it used to be called but the automatic namer turns it into normalize_utf8
// I think that's arguably more correct so we'll support both.
@LegacyNames("normalize_utf_8")
@Data
@EqualsAndHashCode(callSuper = false)
@Accessors(fluent = true, chain = true)
@NoArgsConstructor(access = AccessLevel.PACKAGE)
public class NormalizeUtf8Transform extends StringTransform<NormalizeUtf8Transform> {
  public static final Normalizer.Form DEFAULT_NORMALIZATION_FORM = Normalizer.Form.NFD;

  private String normalizationFormString;

  @Setter(AccessLevel.NONE)
  private Normalizer.Form normalizationForm;

  @Override
  public NormalizeUtf8Transform configure(Config config, String key) {
    return super.configure(config, key)
        .normalizationFormString(stringFromConfig(config, key, ".normalization_form", false));
  }

  @Override
  protected void setup() {
    super.setup();
    if (normalizationFormString == null) {
      normalizationForm = DEFAULT_NORMALIZATION_FORM;
      return;
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
    return Normalizer.normalize(rawString, normalizationForm);
  }
}
