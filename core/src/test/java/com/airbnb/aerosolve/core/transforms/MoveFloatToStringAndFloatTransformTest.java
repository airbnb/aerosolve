package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;

import java.util.HashMap;
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
public class MoveFloatToStringAndFloatTransformTest {
  private static final Logger log = LoggerFactory.getLogger(MoveFloatToStringAndFloatTransformTest.class);

  public String makeConfig() {
    return "test_move_float_to_string_and_float {\n" +
        " transform : move_float_to_string_and_float\n" +
        " field1 : floatFeature1\n" +
        " bucket : 1.0\n" +
        " max_bucket : 10.0\n" +
        " min_bucket : 0.0\n" +
        " string_output : stringOutput\n" +
        " float_output : floatOutput\n" +
        "}";
  }

  public FeatureVector makeFeatureVector() {
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Map<String, Double> floatFeature1 = new HashMap<>();

    floatFeature1.put("a", 0.0);
    floatFeature1.put("b", 10.0);
    floatFeature1.put("c", 9.9);
    floatFeature1.put("d", 10.1);
    floatFeature1.put("e", 11.01);
    floatFeature1.put("f", -0.1);
    floatFeature1.put("g", -1.01);
    floatFeature1.put("h", 21.3);
    floatFeature1.put("i", 2000.0);

    floatFeatures.put("floatFeature1", floatFeature1);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(
        config, "test_move_float_to_string_and_float");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);

    assertTrue(featureVector.getStringFeatures() == null);
    assertTrue(featureVector.getFloatFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(
        config, "test_move_float_to_string_and_float");
    FeatureVector featureVector = makeFeatureVector();

    transform.doTransform(featureVector);

    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();

    assertNotNull(stringFeatures);
    assertEquals(1, stringFeatures.size());

    assertNotNull(floatFeatures);
    assertEquals(2, floatFeatures.size());

    Set<String> stringOutput = stringFeatures.get("stringOutput");
    Map<String, Double> floatOutput = floatFeatures.get("floatOutput");

    assertEquals(5, stringOutput.size());
    assertEquals(4, floatOutput.size());

    assertTrue(stringOutput.contains("a=0.0"));
    assertTrue(stringOutput.contains("b=10.0"));
    assertTrue(stringOutput.contains("c=9.0"));
    assertTrue(stringOutput.contains("d=10.0"));
    assertTrue(stringOutput.contains("f=0.0"));

    assertEquals(11.01, floatOutput.get("e"), 0.0);
    assertEquals(-1.01, floatOutput.get("g"), 0.0);
    assertEquals(21.3, floatOutput.get("h"), 0.0);
    assertEquals(2000.0, floatOutput.get("i"), 0.0);
  }
}
