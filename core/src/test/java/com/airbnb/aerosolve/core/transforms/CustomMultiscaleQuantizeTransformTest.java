package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;

import java.util.HashSet;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertTrue;

public class CustomMultiscaleQuantizeTransformTest {
    private static final Logger log = LoggerFactory.getLogger(CustomMultiscaleQuantizeTransformTest.class);

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
      map.put("zero", 0.0);
      map.put("negative", -1.0);
      floatFeatures.put("loc", map);

      FeatureVector featureVector = new FeatureVector();
      featureVector.setStringFeatures(stringFeatures);
      featureVector.setFloatFeatures(floatFeatures);
      return featureVector;
    }

    public String makeConfig(String input) {
      return "test_quantize {\n" +
          " transform : custom_multiscale_quantize\n" +
          " field1 : loc\n" + input +
          " buckets : [1, 10]\n" +
          " output : loc_quantized\n" +
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
      Set<String> out = stringFeatures.get("loc_quantized");
      log.info("quantize output");
      for (String string : out) {
        log.info(string);
      }
      assertTrue(out.size() == 7);
      assertTrue(out.contains("lat[10.0]=30.0"));
      assertTrue(out.contains("long[1.0]=40.0"));
      assertTrue(out.contains("long[10.0]=40.0"));
      assertTrue(out.contains("lat[1.0]=37.0"));
      assertTrue(out.contains("zero=0"));
      assertTrue(out.contains("negative[1.0]=-1.0"));
      assertTrue(out.contains("negative[10.0]=0.0"));
    }

  @Test
  public void testSelectFeatures() {
    Config config = ConfigFactory.parseString(makeConfig("select_features: [\"lat\"] \n"));
    Transform transform = TransformFactory.createTransform(config, "test_quantize");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 2);
    Set<String> out = stringFeatures.get("loc_quantized");
    log.info("quantize output");
    for (String string : out) {
      log.info(string);
    }
    assertTrue(out.size() == 2);
    assertTrue(out.contains("lat[10.0]=30.0"));
    assertTrue(out.contains("lat[1.0]=37.0"));
  }

  @Test
  public void testExcludeFeatures() {
    Config config = ConfigFactory.parseString(makeConfig("exclude_features: [\"lat\"] \n"));
    Transform transform = TransformFactory.createTransform(config, "test_quantize");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 2);
    Set<String> out = stringFeatures.get("loc_quantized");
    log.info("quantize output");
    for (String string : out) {
      log.info(string);
    }
    assertTrue(out.size() == 5);
    assertTrue(out.contains("long[1.0]=40.0"));
    assertTrue(out.contains("long[10.0]=40.0"));
    assertTrue(out.contains("zero=0"));
    assertTrue(out.contains("negative[1.0]=-1.0"));
    assertTrue(out.contains("negative[10.0]=0.0"));
  }

  @Test
  public void testSelectAndExcludeFeatures() {
    Config config = ConfigFactory.parseString(
        makeConfig("select_features: [\"lat\", \"long\"] \n" + "exclude_features: [\"lat\"] \n"));
    Transform transform = TransformFactory.createTransform(config, "test_quantize");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 2);
    Set<String> out = stringFeatures.get("loc_quantized");
    log.info("quantize output");
    for (String string : out) {
      log.info(string);
    }
    assertTrue(out.size() == 2);
    assertTrue(out.contains("long[1.0]=40.0"));
    assertTrue(out.contains("long[10.0]=40.0"));
  }
}
