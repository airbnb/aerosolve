package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class NormalizeFloatTransformTest {
  private static final Logger log = LoggerFactory.getLogger(CapFloatTransformTest.class);
  
  public String makeConfig() {
    return "test_norm {\n" +
           " transform : normalize_float\n" +
           " field1 : loc\n" +
           "}";
  }

  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_norm");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);
    assertTrue(featureVector.getStringFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_norm");
    FeatureVector featureVector = TransformTestingHelper.makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 1);

    Map<String, Double> feat1 = featureVector.getFloatFeatures().get("loc");

    assertEquals(3, feat1.size());
    double scale = 1.0 / Math.sqrt(37.7 * 37.7 + 40.0 * 40.0 + 20.0 * 20.0);
    assertEquals(scale * 37.7, feat1.get("lat"), 0.1);
    assertEquals(scale * 40.0, feat1.get("long"), 0.1);
    assertEquals(scale * -20.0, feat1.get("z"), 0.1);
  }

}