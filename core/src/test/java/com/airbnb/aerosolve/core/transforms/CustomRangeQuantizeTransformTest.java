package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

import static org.junit.Assert.assertTrue;

public class CustomRangeQuantizeTransformTest {
  private static final Logger log = LoggerFactory.getLogger(CustomRangeQuantizeTransformTest.class);

  private FeatureVector makeFeatureVector() {
    Map<String, Set<String>> stringFeatures = new HashMap<>();
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Set list = new HashSet<String>();
    list.add("aaa");
    list.add("bbb");
    stringFeatures.put("strFeature1", list);

    Map<String, Double> map = new HashMap<>();
    map.put("short", 3.0);
    map.put("med", 8.0);
    map.put("long", 30.0);
    map.put("low_bound", 7.0);
    map.put("high_bound", 28.0);
    floatFeatures.put("trip", map);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  public String makeConfig(String input) {
    return "test_quantize {\n" +
        " transform : custom_range_quantize\n" +
        " field1 : trip\n" + input +
        " thresholds : [7.0, 28.0]\n" +
        " output : trip_quantized\n" +
        "}";
  }

  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig(""));
    Transform transform = TransformFactory.createTransform(config, "test_quantize");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);
    assertTrue(featureVector.getStringFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig(""));
    Transform transform = TransformFactory.createTransform(config, "test_quantize");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 2);
    assertTrue(stringFeatures.keySet().contains("trip_quantized"));
    assertTrue(stringFeatures.keySet().contains("strFeature1"));
    Set<String> out = stringFeatures.get("trip_quantized");
    assertTrue(out.size() == 5);

    assertTrue(out.contains("short<=7.0"));
    assertTrue(out.contains("low_bound<=7.0"));
    assertTrue(out.contains("7.0<med<=28.0"));
    assertTrue(out.contains("long>28.0"));
    assertTrue(out.contains("7.0<high_bound<=28.0"));
  }

  @Test
  public void testSelectFeatures() {
    Config config = ConfigFactory.parseString(makeConfig("select_features: [\"short\"] \n"));
    Transform transform = TransformFactory.createTransform(config, "test_quantize");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 2);
    Set<String> out = stringFeatures.get("trip_quantized");
    assertTrue(out.size() == 1);
    assertTrue(out.contains("short<=7.0"));
  }
}
