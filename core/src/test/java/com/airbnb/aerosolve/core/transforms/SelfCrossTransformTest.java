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

/**
 * @author Hector Yee
 */
public class SelfCrossTransformTest {
  private static final Logger log = LoggerFactory.getLogger(SelfCrossTransformTest.class);

  public FeatureVector makeFeatureVector() {
    HashMap stringFeatures = new HashMap<String, Set<String>>();

    Set list = new HashSet<String>();
    list.add("aaa");
    list.add("bbb");
    stringFeatures.put("feature1", list);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    return featureVector;
  }

  public String makeConfig() {
    return "test_cross {\n" +
           " transform : self_cross\n" +
           " field1 : feature1\n" +
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
    assertTrue(stringFeatures.size() == 2);
    Set<String> out = stringFeatures.get("out");
    log.info("Cross output");
    for (String string : out) {
      log.info(string);
    }
    assertTrue(out.size() == 1);
    assertTrue(out.contains("aaa^bbb"));
  }
}