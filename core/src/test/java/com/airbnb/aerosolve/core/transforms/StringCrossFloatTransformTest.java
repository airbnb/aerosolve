package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Map;
import java.util.HashSet;
import java.util.Set;
import java.util.HashMap;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertEquals;

/**
 * @author Hector Yee
 */
public class StringCrossFloatTransformTest {
  private static final Logger log = LoggerFactory.getLogger(StringCrossFloatTransformTest.class);

  public FeatureVector makeFeatureVector() {
    Map<String, Set<String>> stringFeatures = new HashMap<>();
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Set list = new HashSet<String>();
    list.add("aaa");
    list.add("bbb");
    stringFeatures.put("strFeature1", list);

    Map<String, Double> map = new HashMap<>();
    map.put("lat", 37.7);
    map.put("long", 40.0);
    floatFeatures.put("loc", map);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  public String makeConfig() {
    return "test_cross {\n" +
           " transform : string_cross_float\n" +
           " field1 : strFeature1\n" +
           " field2 : loc\n" +
           " output : out\n" +
           "}";
  }
  
  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_cross");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);
    assertTrue(featureVector.getStringFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_cross");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertEquals(1, stringFeatures.size());
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    assertEquals(2, floatFeatures.size());
    Map<String, Double> out = floatFeatures.get("out");
    assertTrue(out.size() == 4);
    log.info("Cross output");
    for (Map.Entry<String, Double> entry : out.entrySet()) {
      log.info(entry.getKey() + '=' + entry.getValue());
    }
    assertEquals(37.7, out.get("aaa^lat"), 0.1);
    assertEquals(37.7, out.get("bbb^lat"), 0.1);
    assertEquals(40.0, out.get("aaa^long"), 0.1);
    assertEquals(40.0, out.get("bbb^long"), 0.1);
  }
}