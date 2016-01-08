package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Created by seckcoder on 12/17/15.
 */
public class DateDiffTransformTest {
  private static final Logger log = LoggerFactory.getLogger(DateDiffTransformTest.class);
  public String makeConfig() {
    return "test_datediff {\n" +
            " transform: date_diff\n" +
            " field1: endDates\n" +
            " field2: startDates\n" +
            " output: bar\n" +
            "}";
  }

  public FeatureVector makeFeatureVector() {
    Map<String, Set<String>> stringFeatures = new HashMap<>();
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();
    Set endDates = new HashSet<String>();
    Set startDates = new HashSet<String>();
    endDates.add("2009-03-01");
    startDates.add("2009-02-27");
    stringFeatures.put("endDates", endDates);
    stringFeatures.put("startDates", startDates);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_datediff");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);

    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    assertTrue(floatFeatures.size() == 1);

    Map<String, Double> out = floatFeatures.get("bar");
    assertEquals(2, out.get("2009-03-01-m-2009-02-27").intValue());
  }
}
