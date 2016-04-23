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

  public String makeConfig(boolean moveAllKeys) {
    StringBuilder sb = new StringBuilder();
    sb.append("test_move_float_to_string {\n");
    sb.append(" transform : move_float_to_string\n");
    sb.append(" field1 : loc\n");
    sb.append(" bucket : 1\n");

    if (!moveAllKeys) {
      sb.append(" keys : [lat]\n");
    }

    sb.append(" output : loc_quantized\n");
    sb.append("}");
    return sb.toString();
  }
  
  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig(false));
    Transform transform = TransformFactory.createTransform(config, "test_move_float_to_string");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);
    assertTrue(featureVector.getStringFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig(false));
    Transform transform = TransformFactory.createTransform(config, "test_move_float_to_string");
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

  @Test
  public void testTransformMoveAllKeys() {
    Config config = ConfigFactory.parseString(makeConfig(true));
    Transform transform = TransformFactory.createTransform(config, "test_move_float_to_string");
    FeatureVector featureVector = TransformTestingHelper.makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 2);
    Set<String> out = stringFeatures.get("loc_quantized");
    assertTrue(out.size() == 3);
    log.info("quantize output");
    for (String string : out) {
      log.info(string);
    }
    assertTrue(out.contains("lat=37.0"));
    assertTrue(out.contains("long=40.0"));
    assertTrue(out.contains("z=-20.0"));
  }
}