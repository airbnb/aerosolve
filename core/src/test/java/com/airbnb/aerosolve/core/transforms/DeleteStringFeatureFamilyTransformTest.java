package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;

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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * Created by christhetree on 1/29/16.
 */
public class DeleteStringFeatureFamilyTransformTest {
  private static final Logger log = LoggerFactory.getLogger(
      DeleteStringFeatureFamilyTransformTest.class);

  public String makeConfig() {
    return "test_delete_string_feature_family {\n" +
        " transform: delete_string_feature_family\n" +
        " field1: strFeature1\n" +
        "}";
  }

  public FeatureVector makeFeatureVector() {
    Map<String, Set<String>> stringFeatures = new HashMap<>();

    Set list = new HashSet<String>();
    list.add("I am a string in a string feature");
    stringFeatures.put("strFeature1", list);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    return featureVector;
  }

  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(
        config, "test_delete_string_feature_family");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);

    assertTrue(featureVector.getStringFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(
        config, "test_delete_string_feature_family");
    FeatureVector featureVector = makeFeatureVector();
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();

    assertTrue(stringFeatures.containsKey("strFeature1"));
    assertEquals(1, stringFeatures.size());

    transform.doTransform(featureVector);

    assertFalse(stringFeatures.containsKey("strFeature1"));
    assertEquals(0, stringFeatures.size());
  }
}
