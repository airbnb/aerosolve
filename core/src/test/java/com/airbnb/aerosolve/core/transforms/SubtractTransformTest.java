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
public class SubtractTransformTest {
  private static final Logger log = LoggerFactory.getLogger(SubtractTransformTest.class);

  public String makeConfigWithKeys() {
    return "test_subtract {\n" +
        " transform : subtract\n" +
        " field1 : loc\n" +
        " field2 : F\n" +
        " keys : [\"lat\"] \n" +
        " key2 : foo\n" +
        " output : bar\n" +
        "}";
  }
  
  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfigWithKeys());
    Transform transform = TransformFactory.createTransform(config, "test_subtract");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);
    assertTrue(featureVector.getStringFeatures() == null);
  }

  @Test
  public void testTransformWithKeys() {
    Config config = ConfigFactory.parseString(makeConfigWithKeys());
    Transform transform = TransformFactory.createTransform(config, "test_subtract");
    FeatureVector featureVector = TransformTestingHelper.makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 1);

    Map<String, Double> out = featureVector.floatFeatures.get("bar");
    for (Map.Entry<String, Double> entry : out.entrySet()) {
      log.info(entry.getKey() + "=" + entry.getValue());
    }
    assertTrue(out.size() == 2);
    assertEquals(36.2, out.get("lat-foo"), 0.1);
    assertEquals(1.0, out.get("bar_fv"), 0.1);
  }
}