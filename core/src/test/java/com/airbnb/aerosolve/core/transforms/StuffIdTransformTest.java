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
public class StuffIdTransformTest {
  private static final Logger log = LoggerFactory.getLogger(StuffIdTransformTest.class);

  public FeatureVector makeFeatureVector() {
    Map<String, Set<String>> stringFeatures = new HashMap<>();
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Set list = new HashSet<String>();
    list.add("aaa");
    list.add("bbb");
    stringFeatures.put("strFeature1", list);

    Map<String, Double> map = new HashMap<>();
    map.put("searches", 37.7);
    floatFeatures.put("FEAT", map);

    Map<String, Double> map2 = new HashMap<>();
    map2.put("id", 123456789.0);
    floatFeatures.put("ID", map2);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  public String makeConfig() {
    return "test_stuff {\n" +
           " transform : stuff_id\n" +
           " field1 : ID\n" +
           " key1 : id\n" +
           " field2 : FEAT\n" +
           " key2 : searches\n" +
           " output : bar\n" +
           "}";
  }
  
  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_stuff");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);
    assertTrue(featureVector.getStringFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_stuff");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 1);

    log.info(featureVector.toString());

    Map<String, Double> out = featureVector.floatFeatures.get("bar");
    for (Map.Entry<String, Double> entry : out.entrySet()) {
      log.info(entry.getKey() + "=" + entry.getValue());
    }
    assertTrue(out.size() == 1);
    assertEquals(37.7, out.get("searches@123456789"), 0.1);
  }
}