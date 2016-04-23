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
import static org.junit.Assert.assertNotNull;
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
        " fields: [strFeature1, strFeature2, strFeature3]\n" +
        "}";
  }

  public FeatureVector makeFeatureVector() {
    Map<String, Set<String>> stringFeatures = new HashMap<>();

    Set<String> list1 = new HashSet<>();
    list1.add("I am a string in string feature 1");
    stringFeatures.put("strFeature1", list1);

    Set<String> list2 = new HashSet<>();
    list2.add("I am a string in string feature 2");
    stringFeatures.put("strFeature2", list2);

    Set<String> list3 = new HashSet<>();
    list3.add("I am a string in string feature 3");
    stringFeatures.put("strFeature3", list3);

    Set<String> list4 = new HashSet<>();
    list4.add("I am a string in string feature 4");
    stringFeatures.put("strFeature4", list4);

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

    assertNotNull(stringFeatures);
    assertTrue(stringFeatures.containsKey("strFeature1"));
    assertTrue(stringFeatures.containsKey("strFeature2"));
    assertTrue(stringFeatures.containsKey("strFeature3"));
    assertTrue(stringFeatures.containsKey("strFeature4"));
    assertEquals(4, stringFeatures.size());

    transform.doTransform(featureVector);

    assertNotNull(stringFeatures);
    assertFalse(stringFeatures.containsKey("strFeature1"));
    assertFalse(stringFeatures.containsKey("strFeature2"));
    assertFalse(stringFeatures.containsKey("strFeature3"));
    assertTrue(stringFeatures.containsKey("strFeature4"));
    assertEquals(1, stringFeatures.size());
  }
}
