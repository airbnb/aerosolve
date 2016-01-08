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
public class MoveFloatToStringTransformTest {
  private static final Logger log = LoggerFactory.getLogger(MoveFloatToStringTransformTest.class);

  public String makeConfig() {
    return "test_quantize {\n" +
           " transform : move_float_to_string\n" +
           " field1 : loc\n" +
           " bucket : 1\n" +
           " keys : [lat]\n" +
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
    FeatureVector featureVector = TransformTestingHelper.makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 2);
    Set<String> out = stringFeatures.get("loc_quantized");
    assertTrue(out.size() == 1);
    log.info("quantize output");
    for (String string : out) {
      log.info(string);
    }
    assertTrue(out.contains("lat=37.0"));
  }
}