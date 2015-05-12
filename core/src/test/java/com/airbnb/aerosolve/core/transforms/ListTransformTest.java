package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashSet;
import java.util.Map;
import java.util.List;
import java.util.HashMap;
import java.util.Set;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class ListTransformTest {
  private static final Logger log = LoggerFactory.getLogger(ListTransformTest.class);

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
    return "test_quantize {\n" +
           " transform : quantize\n" +
           " field1 : loc\n" +
           " scale : 10\n" +
           " output : loc_quantized\n" +
           "}\n" +
           "test_cross {\n" +
           " transform : cross\n" +
           " field1 : strFeature1\n" +
           " field2 : loc_quantized\n" +
           " output : out\n" +
           "}\n" +
           "test_list {\n" +
           " transform : list\n" +
           " transforms : [test_quantize, test_cross]\n" +
           "}";
  }
  
  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_list");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);
    assertTrue(featureVector.getStringFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_list");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 3);
    Set<String> out = stringFeatures.get("out");
    assertTrue(out.size() == 4);
    log.info("crossed quantized output");
    for (String string : out) {
      log.info(string);
    }
    assertTrue(out.contains("bbb^long=400"));
    assertTrue(out.contains("aaa^lat=377"));
  }
}