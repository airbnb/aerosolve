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
public class KdtreeContinuousTransformTest {
  private static final Logger log = LoggerFactory.getLogger(KdtreeContinuousTransformTest.class);

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
    return "test_kdtree {\n" +
           " transform : kdtree_continuous\n" +
           " include \"test_kdt.model.conf\"\n" +
           " field1 : loc\n" +
           " value1 : lat\n" +
           " value2 : long\n" +
           " max_count : 3\n" +
           " output : loc_kdt\n" +
           "}";
  }
  
  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_kdtree");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);
    assertTrue(featureVector.getStringFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    log.info("Model encoded is " + config.getString("test_kdtree.model_base64"));
    Transform transform = TransformFactory.createTransform(config, "test_kdtree");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 1);
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    Map<String, Double> out = floatFeatures.get("loc_kdt");
    log.info("loc_kdt");
    for (Map.Entry<String, Double> entry : out.entrySet()) {
      log.info(entry.getKey() + " = " + entry.getValue());
    }
    assertTrue(out.size() == 2);
    //                    4
    //         |--------------- y = 2
    //  1      | 2       3
    //     x = 1
    assertEquals(out.get("0"), 37.7 - 1.0, 0.1);
    assertEquals(out.get("2"), 40.0 - 2.0, 0.1);
  }
}