package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

public class FloatToStringTransformTest {
  public String makeConfig() {
    return "test_float_to_string_and_float {\n" +
        " transform : float_to_string\n" +
        " field1 : floatFeature1\n" +
        " keys : [a, b, g]\n" +
        " values : [0.0, 10.0]\n" +
        " string_output : stringOutput\n" +
        "}";
  }

  public FeatureVector makeFeatureVector() {
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Map<String, Double> floatFeature1 = new HashMap<>();

    floatFeature1.put("a", 0.0);
    floatFeature1.put("b", 10.0);
    floatFeature1.put("c", 21.3);
    floatFeature1.put("d", 10.1);
    floatFeature1.put("e", 11.01);
    floatFeature1.put("f", -1.01);
    floatFeature1.put("g", 0d);

    floatFeatures.put("floatFeature1", floatFeature1);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(
        config, "test_float_to_string_and_float");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);

    assertTrue(featureVector.getStringFeatures() == null);
    assertTrue(featureVector.getFloatFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(
        config, "test_float_to_string_and_float");
    FeatureVector featureVector = makeFeatureVector();

    transform.doTransform(featureVector);

    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();

    assertNotNull(stringFeatures);
    assertEquals(1, stringFeatures.size());

    assertNotNull(floatFeatures);
    assertEquals(1, floatFeatures.size());

    Set<String> stringOutput = stringFeatures.get("stringOutput");
    Map<String, Double> floatOutput = floatFeatures.get("floatFeature1");

    assertEquals(3, stringOutput.size());
    assertEquals(4, floatOutput.size());

    assertTrue(stringOutput.contains("a=0.0"));
    assertTrue(stringOutput.contains("b=10.0"));
    assertTrue(stringOutput.contains("g=0.0"));

    assertEquals(21.3, floatOutput.get("c"), 0.0);
    assertEquals(10.1, floatOutput.get("d"), 0.0);
    assertEquals(11.01, floatOutput.get("e"), 0.0);
    assertEquals(-1.01, floatOutput.get("f"), 0.0);
  }
}
