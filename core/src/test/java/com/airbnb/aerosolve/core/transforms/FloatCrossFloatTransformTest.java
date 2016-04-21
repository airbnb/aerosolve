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
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Created by christhetree on 4/8/16.
 */
public class FloatCrossFloatTransformTest {
  private static final Logger log = LoggerFactory.getLogger(FloatCrossFloatTransformTest.class);

  public String makeConfig() {
    return "test_float_cross_float {\n" +
        " transform : float_cross_float\n" +
        " field1 : floatFeature1\n" +
        " bucket : 1.0\n" +
        " cap : 1000.0\n" +
        " field2 : floatFeature2\n" +
        " output : out\n" +
        "}";
  }

  public FeatureVector makeFeatureVector() {
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Map<String, Double> floatFeature1 = new HashMap<>();

    floatFeature1.put("x", 50.0);
    floatFeature1.put("y", 1.3);
    floatFeature1.put("z", 2000.0);

    Map<String, Double> floatFeature2 = new HashMap<>();

    floatFeature2.put("i", 1.2);
    floatFeature2.put("j", 3.4);
    floatFeature2.put("k", 5.6);

    floatFeatures.put("floatFeature1", floatFeature1);
    floatFeatures.put("floatFeature2", floatFeature2);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_float_cross_float");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);

    assertTrue(featureVector.getFloatFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_float_cross_float");
    FeatureVector featureVector = makeFeatureVector();

    transform.doTransform(featureVector);

    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();

    assertNotNull(floatFeatures);
    assertEquals(3, floatFeatures.size());

    Map<String, Double> out = floatFeatures.get("out");

    assertEquals(9, out.size());

    assertEquals(1.2, out.get("x=50.0^i"), 0.0);
    assertEquals(1.2, out.get("y=1.0^i"), 0.0);
    assertEquals(1.2, out.get("z=1000.0^i"), 0.0);
    assertEquals(3.4, out.get("x=50.0^j"), 0.0);
    assertEquals(3.4, out.get("y=1.0^j"), 0.0);
    assertEquals(3.4, out.get("z=1000.0^j"), 0.0);
    assertEquals(5.6, out.get("x=50.0^k"), 0.0);
    assertEquals(5.6, out.get("y=1.0^k"), 0.0);
    assertEquals(5.6, out.get("z=1000.0^k"), 0.0);
  }
}
