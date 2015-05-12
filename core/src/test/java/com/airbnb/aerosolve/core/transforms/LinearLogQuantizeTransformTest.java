package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class LinearLogQuantizeTransformTest {
  private static final Logger log = LoggerFactory.getLogger(LinearLogQuantizeTransformTest.class);

  public FeatureVector makeFeatureVector() {
    Map<String, Set<String>> stringFeatures = new HashMap<>();
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Set list = new HashSet<String>();
    list.add("aaa");
    list.add("bbb");
    stringFeatures.put("strFeature1", list);

    Map<String, Double> map = new HashMap<>();
    map.put("a", 0.0);
    map.put("b", 0.13);
    map.put("c", 1.23);
    map.put("d", 5.0);
    map.put("e", 17.5);
    map.put("f", 99.98);
    map.put("g", 365.0);
    map.put("h", 65537.0);
    map.put("i", -1.0);
    map.put("j", -23.0);

    floatFeatures.put("loc", map);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  public String makeConfig() {
    return "test_quantize {\n" +
           " transform : linear_log_quantize\n" +
           " field1 : loc\n" +
           " scale : 10\n" +
           " output : loc_quantized\n" +
           "}";
  }
  
  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_quantize");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);
    assertTrue(featureVector.getStringFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_quantize");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 2);
    Set<String> out = stringFeatures.get("loc_quantized");
    assertTrue(out.size() == 10);
    log.info("quantize output");
    for (String string : out) {
      log.info(string);
    }
    assertTrue(out.contains("a=0"));
    assertTrue(out.contains("b=0.125"));
    assertTrue(out.contains("c=1.125"));
    assertTrue(out.contains("d=5.0"));
    assertTrue(out.contains("e=17.5"));
    assertTrue(out.contains("f=90"));
    assertTrue(out.contains("g=350"));
    assertTrue(out.contains("h=65536"));
    assertTrue(out.contains("i=-1.0"));
    assertTrue(out.contains("j=-23.0"));
  }
}