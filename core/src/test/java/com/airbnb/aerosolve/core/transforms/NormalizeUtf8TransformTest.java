package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import java.text.Normalizer;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * Created by christhetree on 1/27/16.
 */
public class NormalizeUtf8TransformTest extends BaseTransformTest {
  public static final String FUNKY_STRING = "Funky string: \u03D3\u03D4\u1E9B";

  public String makeConfigWithoutNormalizationFormAndOutput() {
    return "test_normalize_utf_8 {\n" +
        " transform: normalize_utf_8\n" +
        " field1: strFeature1\n" +
        "}";
  }

  public String makeConfig() {
    return makeConfigWithoutNormalizationFormAndOutput();
  }

  public String makeConfigWithNormalizationForm(String normalizationForm) {
    return "test_normalize_utf_8 {\n" +
        " transform: normalize_utf_8\n" +
        " field1: strFeature1\n" +
        " normalization_form: " + normalizationForm + "\n" +
        " output: bar\n" +
        "}";
  }

  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .string("strFeature1", FUNKY_STRING)
        .build();
  }

  @Override
  public String configKey() {
    return "test_normalize_utf_8";
  }

  @Test
  public void testTransformDefaultNormalizationFormAndOverwriteInput() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 1);

    assertStringFamily(featureVector, "strFeature1", 1, ImmutableSet.of(
        Normalizer.normalize(FUNKY_STRING, NormalizeUtf8Transform.DEFAULT_NORMALIZATION_FORM)
    ));
  }

  @Test
  public void testTransformNfcNormalizationForm() {
    Transform<MultiFamilyVector> transform = getTransform(
        makeConfigWithNormalizationForm("NFC"), configKey());
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 2);

    assertStringFamily(featureVector, "bar", 1, ImmutableSet.of(FUNKY_STRING));
    assertTrue(Normalizer.isNormalized(FUNKY_STRING, Normalizer.Form.NFC));
  }

  @Test
  public void testTransformNfdNormalizationForm() {
    Transform<MultiFamilyVector> transform = getTransform(
        makeConfigWithNormalizationForm("NFD"), configKey());
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 2);

    String normalizedString = "Funky string: \u03D2\u0301\u03D2\u0308\u017F\u0307";
    assertStringFamily(featureVector, "bar", 1, ImmutableSet.of(normalizedString));
    assertTrue(Normalizer.isNormalized(normalizedString, Normalizer.Form.NFD));
  }

  @Test
  public void testTransformNfkcNormalizationForm() {
    Transform<MultiFamilyVector> transform = getTransform(
        makeConfigWithNormalizationForm("NFKC"), configKey());
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 2);

    String normalizedString = "Funky string: \u038e\u03ab\u1e61";
    assertStringFamily(featureVector, "bar", 1, ImmutableSet.of(normalizedString));
    assertTrue(Normalizer.isNormalized(normalizedString, Normalizer.Form.NFKC));
  }

  @Test
  public void testTransformNfkdNormalizationForm() {
    Transform<MultiFamilyVector> transform = getTransform(makeConfigWithNormalizationForm("NFKD"), configKey());
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 2);

    String normalizedString = "Funky string: \u03a5\u0301\u03a5\u0308\u0073\u0307";
    assertStringFamily(featureVector, "bar", 1, ImmutableSet.of(normalizedString));
    assertTrue(Normalizer.isNormalized(normalizedString, Normalizer.Form.NFKD));
  }
}
