package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class CrossTransformTest {
  private static final Logger log = LoggerFactory.getLogger(CrossTransformTest.class);

  public FeatureVector makeFeatureVector() {
    HashMap stringFeatures = new HashMap<String, Set<String>>();

    Set list = new HashSet<String>();
    list.add("aaa");
    list.add("bbb");
    stringFeatures.put("feature1", list);

    Set list2 = new HashSet<String>();
    list2.add("11");
    list2.add("22");
    stringFeatures.put("feature2", list2);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    return featureVector;
  }

  public String makeConfig() {
    return "test_cross {\n" +
           " transform : cross\n" +
           " field1 : feature1\n" +
           " field2 : feature2\n" +
           " output : out\n" +
           "}";
  }

  public String makeKey1Config() {
    return "test_cross {\n" +
        " transform : cross\n" +
        " field1 : feature1\n" +
        " keys1 : [aaa]\n" +
        " field2 : feature2\n" +
        " output : out\n" +
        "}";
  }

  public String makeBothKeyConfig() {
    return "test_cross {\n" +
        " transform : cross\n" +
        " field1 : feature1\n" +
        " keys1 : [aaa]\n" +
        " field2 : feature2\n" +
        " keys2 : [22]\n" +
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
    assertTrue(stringFeatures.size() == 3);
    Set<String> out = stringFeatures.get("out");
    assertTrue(out.size() == 4);
    log.info("Cross output");
    for (String string : out) {
      log.info(string);
    }
    assertTrue(out.contains("aaa^11"));
    assertTrue(out.contains("aaa^22"));
    assertTrue(out.contains("bbb^11"));
    assertTrue(out.contains("bbb^22"));
  }

  @Test
  public void testOneKeyTransform() {
    Config config = ConfigFactory.parseString(makeKey1Config());
    Transform transform = TransformFactory.createTransform(config, "test_cross");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 3);
    Set<String> out = stringFeatures.get("out");
    log.info("Cross output");
    for (String string : out) {
      log.info(string);
    }
    assertEquals(2, out.size());
    assertTrue(out.contains("aaa^11"));
    assertTrue(out.contains("aaa^22"));
  }

  @Test
  public void testTwoKeysTransform() {
    Config config = ConfigFactory.parseString(makeBothKeyConfig());
    Transform transform = TransformFactory.createTransform(config, "test_cross");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 3);
    Set<String> out = stringFeatures.get("out");
    log.info("Cross output");
    for (String string : out) {
      log.info(string);
    }
    assertEquals(1, out.size());
    assertTrue(out.contains("aaa^22"));
  }

}
