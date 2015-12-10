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
public class DivideTransformTest {
  private static final Logger log = LoggerFactory.getLogger(DivideTransformTest.class);

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
    map.put("z", -1.0);
    floatFeatures.put("loc", map);

    Map<String, Double> map2 = new HashMap<>();
    map2.put("foo", 1.5);
    floatFeatures.put("F", map2);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  public String makeConfig() {
    return "test_divide {\n" +
        " transform : divide\n" +
        " field1 : loc\n" +
        " field2 : F\n" +
        " key2 : foo\n" +
        " constant : 0.1\n" +
        " output : bar\n" +
        "}";
  }

  public String makeConfigWithKeys() {
    return "test_divide {\n" +
           " transform : divide\n" +
           " field1 : loc\n" +
           " field2 : F\n" +
           " keys : [ lat, long ] \n" +
           " key2 : foo\n" +
           " constant : 0.1\n" +
           " output : bar\n" +
           "}";
  }
  
  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_divide");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);
    assertTrue(featureVector.getStringFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_divide");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 1);

    Map<String, Double> out = featureVector.floatFeatures.get("bar");
    for (Map.Entry<String, Double> entry : out.entrySet()) {
      log.info(entry.getKey() + "=" + entry.getValue());
    }
    assertTrue(out.size() == 3);
    assertEquals(37.7 / 1.6, out.get("lat-d-foo"), 0.1);
    assertEquals(40.0 / 1.6, out.get("long-d-foo"), 0.1);
    assertEquals(-1.0 / 1.6, out.get("z-d-foo"), 0.1);
  }

  @Test
  public void testTransformWithKeys() {
    Config config = ConfigFactory.parseString(makeConfigWithKeys());
    Transform transform = TransformFactory.createTransform(config, "test_divide");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 1);

    Map<String, Double> out = featureVector.floatFeatures.get("bar");
    for (Map.Entry<String, Double> entry : out.entrySet()) {
      log.info(entry.getKey() + "=" + entry.getValue());
    }
    assertTrue(out.size() == 2);
    assertEquals(37.7 / 1.6, out.get("lat-d-foo"), 0.1);
    assertEquals(40.0 / 1.6, out.get("long-d-foo"), 0.1);
  }
}