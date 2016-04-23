package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;

import java.text.Normalizer;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Created by christhetree on 1/27/16.
 */
public class NormalizeUtf8TransformTest {
  private static final Logger log = LoggerFactory.getLogger(NormalizeUtf8TransformTest.class);

  public String makeConfigWithoutNormalizationFormAndOutput() {
    return "test_normalize_utf_8 {\n" +
        " transform: normalize_utf_8\n" +
        " field1: strFeature1\n" +
        "}";
  }

  public String makeConfigWithNormalizationForm(String normalizationForm) {
    return "test_normalize_utf_8 {\n" +
        " transform: normalize_utf_8\n" +
        " field1: strFeature1\n" +
        " normalization_form: " + normalizationForm + "\n" +
        " output: bar\n" +
        "}";
  }

  public FeatureVector makeFeatureVector() {
    Map<String, Set<String>> stringFeatures = new HashMap<>();

    Set<String> list = new HashSet<>();
    list.add("Funky string: \u03D3\u03D4\u1E9B");
    stringFeatures.put("strFeature1", list);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    return featureVector;
  }

  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfigWithoutNormalizationFormAndOutput());
    Transform transform = TransformFactory.createTransform(config, "test_normalize_utf_8");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);

    assertTrue(featureVector.getStringFeatures() == null);
  }

  @Test
  public void testTransformDefaultNormalizationFormAndOverwriteInput() {
    Config config = ConfigFactory.parseString(makeConfigWithoutNormalizationFormAndOutput());
    Transform transform = TransformFactory.createTransform(config, "test_normalize_utf_8");
    FeatureVector featureVector = makeFeatureVector();
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    transform.doTransform(featureVector);

    assertNotNull(stringFeatures);
    assertEquals(1, stringFeatures.size());

    Set<String> output = stringFeatures.get("strFeature1");

    assertNotNull(output);
    assertEquals(1, output.size());
    assertTrue(output.contains(Normalizer.normalize(
        "Funky string: \u03D3\u03D4\u1E9B", NormalizeUtf8Transform.DEFAULT_NORMALIZATION_FORM)));
  }

  @Test
  public void testTransformNfcNormalizationForm() {
    Config config = ConfigFactory.parseString(makeConfigWithNormalizationForm("NFC"));
    Transform transform = TransformFactory.createTransform(config, "test_normalize_utf_8");
    FeatureVector featureVector = makeFeatureVector();
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    transform.doTransform(featureVector);

    assertNotNull(stringFeatures);
    assertEquals(2, stringFeatures.size());

    Set<String> output = stringFeatures.get("bar");

    assertNotNull(output);
    assertEquals(1, output.size());
    assertTrue(output.contains("Funky string: \u03D3\u03D4\u1E9B"));
    assertTrue(Normalizer.isNormalized("Funky string: \u03D3\u03D4\u1E9B", Normalizer.Form.NFC));
  }

  @Test
  public void testTransformNfdNormalizationForm() {
    Config config = ConfigFactory.parseString(makeConfigWithNormalizationForm("NFD"));
    Transform transform = TransformFactory.createTransform(config, "test_normalize_utf_8");
    FeatureVector featureVector = makeFeatureVector();
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    transform.doTransform(featureVector);

    assertNotNull(stringFeatures);
    assertEquals(2, stringFeatures.size());

    Set<String> output = stringFeatures.get("bar");

    assertNotNull(output);
    assertEquals(1, output.size());
    assertTrue(output.contains("Funky string: \u03D2\u0301\u03D2\u0308\u017F\u0307"));
    assertTrue(Normalizer.isNormalized(
        "Funky string: \u03D2\u0301\u03D2\u0308\u017F\u0307", Normalizer.Form.NFD));
  }

  @Test
  public void testTransformNfkcNormalizationForm() {
    Config config = ConfigFactory.parseString(makeConfigWithNormalizationForm("NFKC"));
    Transform transform = TransformFactory.createTransform(config, "test_normalize_utf_8");
    FeatureVector featureVector = makeFeatureVector();
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    transform.doTransform(featureVector);

    assertNotNull(stringFeatures);
    assertEquals(2, stringFeatures.size());

    Set<String> output = stringFeatures.get("bar");

    assertNotNull(output);
    assertEquals(1, output.size());
    assertTrue(output.contains("Funky string: \u038e\u03ab\u1e61"));
    assertTrue(Normalizer.isNormalized("Funky string: \u038e\u03ab\u1e61", Normalizer.Form.NFKC));
  }

  @Test
  public void testTransformNfkdNormalizationForm() {
    Config config = ConfigFactory.parseString(makeConfigWithNormalizationForm("NFKD"));
    Transform transform = TransformFactory.createTransform(config, "test_normalize_utf_8");
    FeatureVector featureVector = makeFeatureVector();
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    transform.doTransform(featureVector);

    assertNotNull(stringFeatures);
    assertEquals(2, stringFeatures.size());

    Set<String> output = stringFeatures.get("bar");

    assertNotNull(output);
    assertEquals(1, output.size());
    assertTrue(output.contains("Funky string: \u03a5\u0301\u03a5\u0308\u0073\u0307"));
    assertTrue(Normalizer.isNormalized(
        "Funky string: \u03a5\u0301\u03a5\u0308\u0073\u0307", Normalizer.Form.NFKD));
  }
}
