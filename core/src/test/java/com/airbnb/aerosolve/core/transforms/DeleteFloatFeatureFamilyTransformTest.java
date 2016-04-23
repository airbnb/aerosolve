package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;

import java.util.HashMap;
import java.util.Map;

import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class DeleteFloatFeatureFamilyTransformTest {
  private static final Logger log = LoggerFactory.getLogger(
      DeleteFloatFeatureFamilyTransformTest.class);

  public String makeConfig() {
    return "test_delete_float_feature_family {\n" +
        " transform: delete_float_feature_family\n" +
        " fields: [F1, F2, F3]" +
        "}";
  }

  public FeatureVector makeFeatureVector() {
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Map<String, Double> family1 = new HashMap<>();
    family1.put("A", 1.0);
    family1.put("B", 2.0);

    Map<String, Double> family2 = new HashMap<>();
    family2.put("C", 3.0);
    family2.put("D", 4.0);

    Map<String, Double> family3 = new HashMap<>();
    family3.put("E", 5.0);
    family3.put("F", 6.0);

    Map<String, Double> family4 = new HashMap<>();
    family4.put("G", 7.0);
    family4.put("H", 8.0);

    floatFeatures.put("F1", family1);
    floatFeatures.put("F2", family2);
    floatFeatures.put("F3", family3);
    floatFeatures.put("F4", family4);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(
        config, "test_delete_float_feature_family");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);

    assertTrue(featureVector.getFloatFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(
        config, "test_delete_float_feature_family");
    FeatureVector fv = makeFeatureVector();

    assertNotNull(fv.getFloatFeatures());
    assertTrue(fv.getFloatFeatures().containsKey("F1"));
    assertTrue(fv.getFloatFeatures().containsKey("F2"));
    assertTrue(fv.getFloatFeatures().containsKey("F3"));
    assertTrue(fv.getFloatFeatures().containsKey("F4"));
    assertEquals(4, fv.getFloatFeatures().size());

    transform.doTransform(fv);

    assertNotNull(fv.getFloatFeatures());
    assertFalse(fv.getFloatFeatures().containsKey("F1"));
    assertFalse(fv.getFloatFeatures().containsKey("F2"));
    assertFalse(fv.getFloatFeatures().containsKey("F3"));
    assertTrue(fv.getFloatFeatures().containsKey("F4"));
    assertEquals(1, fv.getFloatFeatures().size());
  }
}
