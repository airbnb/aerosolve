package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;


/**
 * Created by seckcoder on 12/20/15.
 */
public class DateValTransformTest {
  public String makeConfig(String dateType) {
    return "test_date {\n" +
            " transform: date_val\n" +
            " field1: dates\n" +
            " date_type: " + dateType + "\n" +
            " output: bar\n" +
            "}";
  }
  public FeatureVector makeFeatureVector() {
    Map<String, Set<String>> stringFeatures = new HashMap<>();
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Set dates = new HashSet<String>();
    dates.add("2009-03-01");
    dates.add("2009-02-27");
    stringFeatures.put("dates", dates);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  @Test
  public void testDayOfMonthTransform() {
  Config config = ConfigFactory.parseString(makeConfig("day_of_month"));
    Transform transform = TransformFactory.createTransform(config, "test_date");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);

    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    assertTrue(floatFeatures.size() == 1);

    Map<String, Double> out = floatFeatures.get("bar");

    assertEquals(out.get("2009-03-01"), 1, 0.1);
    assertEquals(out.get("2009-02-27"), 27, 0.1);
  }

  @Test
  public void testDayOfWeekTransform() {
    Config config = ConfigFactory.parseString(makeConfig("day_of_week"));
    Transform transform = TransformFactory.createTransform(config, "test_date");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);

    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    assertTrue(floatFeatures.size() == 1);

    Map<String, Double> out = floatFeatures.get("bar");

    assertEquals(out.get("2009-03-01"), 1, 0.1);
    assertEquals(out.get("2009-02-27"), 6, 0.1);
  }

  @Test
  public void testDayOfYearTransform() {
    Config config = ConfigFactory.parseString(makeConfig("day_of_year"));
    Transform transform = TransformFactory.createTransform(config, "test_date");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);

    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    assertTrue(floatFeatures.size() == 1);

    Map<String, Double> out = floatFeatures.get("bar");

    assertEquals(out.get("2009-03-01"), 60, 0.1);
    assertEquals(out.get("2009-02-27"), 58, 0.1);
  }

  @Test
  public void testYearOfDateTransform() {
    Config config = ConfigFactory.parseString(makeConfig("year"));
    Transform transform = TransformFactory.createTransform(config, "test_date");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);

    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    assertTrue(floatFeatures.size() == 1);

    Map<String, Double> out = floatFeatures.get("bar");

    assertEquals(out.get("2009-03-01"), 2009, 0.1);
    assertEquals(out.get("2009-02-27"), 2009, 0.1);
  }

  @Test
  public void testMonthOfDateTransform() {
    Config config = ConfigFactory.parseString(makeConfig("month"));
    Transform transform = TransformFactory.createTransform(config, "test_date");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);

    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    assertTrue(floatFeatures.size() == 1);

    Map<String, Double> out = floatFeatures.get("bar");

    assertEquals(out.get("2009-03-01"), 3, 0.1);
    assertEquals(out.get("2009-02-27"), 2, 0.1);
  }
}
