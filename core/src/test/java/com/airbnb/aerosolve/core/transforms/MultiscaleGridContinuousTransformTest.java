package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertEquals;

/**
 * @author Hector Yee
 */
public class MultiscaleGridContinuousTransformTest {
  private static final Logger log = LoggerFactory.getLogger(MultiscaleGridContinuousTransformTest.class);

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
    return "test_grid {\n" +
           " transform : multiscale_grid_continuous\n" +
           " field1 : loc\n" +
           " value1 : lat\n" +
           " value2 : long\n" +
           " buckets : [1, 5]\n" +
           " output : loc_continuous\n" +
           "}";
  }
  
  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_grid");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);
    assertTrue(featureVector.getStringFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_grid");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 1);
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    assertTrue(floatFeatures.size() == 2);
    Map<String, Double> out = floatFeatures.get("loc_continuous");
    log.info("grid output");
    for (Map.Entry<String, Double> entry : out.entrySet()) {
      log.info(entry.getKey() + "=" + entry.getValue());
    }
    assertEquals(4, out.size());
    assertEquals(0.7, out.get("[1.0]=(37.0,40.0)@1"), 0.01);
    assertEquals(2.7, out.get("[5.0]=(35.0,40.0)@1"), 0.01);
    assertEquals(0.0, out.get("[1.0]=(37.0,40.0)@2"), 0.01);
    assertEquals(0.0, out.get("[5.0]=(35.0,40.0)@2"), 0.01);
  }
}