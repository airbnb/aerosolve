package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by christhetree on 1/27/16.
 */
public class ConvertStringCaseTransformTest {
  private static final Logger log = LoggerFactory.getLogger(ConvertStringCaseTransformTest.class);

  public String makeConfig(boolean convertToUppercase) {
    return "test_convert_string_case {\n" +
        " transform: convert_string_case\n" +
        " field1: strFeature1\n" +
        " convert_to_uppercase: " + Boolean.toString(convertToUppercase) + "\n" +
        " output: bar\n" +
        "}";
  }

  public FeatureVector makeFeatureVector() {
    Map<String, Set<String>> stringFeatures = new HashMap<>();

    Set list = new HashSet<String>();
    list.add("I like BLUEBERRY pie, APPLE pie; and I also like BLUE!");
    list.add("I'm so  excited: I   like blue!?!!");
    stringFeatures.put("strFeature1", list);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    return featureVector;
  }

  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig(false));
    Transform transform = TransformFactory.createTransform(config, "test_convert_string_case");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);

    assertTrue(featureVector.getStringFeatures() == null);
  }

  @Test
  public void testTransformConvertToLowercase() {
    Config config = ConfigFactory.parseString(makeConfig(false));
    Transform transform = TransformFactory.createTransform(config, "test_convert_string_case");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();

    assertEquals(2, stringFeatures.size());

    Set<String> output = stringFeatures.get("bar");

    assertTrue(output.contains("i like blueberry pie, apple pie; and i also like blue!"));
    assertTrue(output.contains("i'm so  excited: i   like blue!?!!"));
  }

  @Test
  public void testTransformConvertToUppercase() {
    Config config = ConfigFactory.parseString(makeConfig(true));
    Transform transform = TransformFactory.createTransform(config, "test_convert_string_case");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();

    assertEquals(2, stringFeatures.size());

    Set<String> output = stringFeatures.get("bar");

    assertTrue(output.contains("I LIKE BLUEBERRY PIE, APPLE PIE; AND I ALSO LIKE BLUE!"));
    assertTrue(output.contains("I'M SO  EXCITED: I   LIKE BLUE!?!!"));
  }
}
