package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 * Created by christhetree on 1/27/16.
 */
public class ReplaceAllStringsTransformTest {
  private static final Logger log = LoggerFactory.getLogger(ReplaceAllStringsTransformTest.class);

  public String makeConfig(List<Map<String, String>> replacements, boolean overwriteInput) {
    StringBuilder sb = new StringBuilder();
    sb.append("test_replace_all_strings {\n");
    sb.append(" transform: replace_all_strings\n");
    sb.append(" field1: strFeature1\n");
    sb.append(" replacements: [\n");
    for (Map<String, String> replacementMap : replacements) {
      for (Map.Entry<String, String> replacementEntry : replacementMap.entrySet()) {
        sb.append("{ \"");
        sb.append(replacementEntry.getKey());
        sb.append("\": \"");
        sb.append(replacementEntry.getValue());
        sb.append("\" }\n");
      }
    }
    sb.append(" ]\n");

    if (!overwriteInput) {
      sb.append(" output: bar\n");
    }

    sb.append("}");
    return sb.toString();
  }

  public FeatureVector makeFeatureVector() {
    Map<String, Set<String>> stringFeatures = new HashMap<>();

    Set<String> list = new HashSet<>();
    list.add("I like blueberry pie, apple pie; and I also like blue!");
    list.add("I'm so  excited: I   like blue!?!!");
    stringFeatures.put("strFeature1", list);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    return featureVector;
  }

  public List<Map<String, String>> makeReplacements() {
    List<Map<String, String>> replacements = new ArrayList<>();
    Map<String, String> replacement1 = new HashMap<>();
    replacement1.put("I ", "you ");
    Map<String, String> replacement2 = new HashMap<>();
    replacement2.put("blue", "yellow");
    Map<String, String> replacement3 = new HashMap<>();
    replacement3.put("yellow", "black");
    replacements.add(replacement1);
    replacements.add(replacement2);
    replacements.add(replacement3);
    return replacements;
  }

  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig(makeReplacements(), false));
    Transform transform = TransformFactory.createTransform(config, "test_replace_all_strings");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);

    assertTrue(featureVector.getStringFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig(makeReplacements(), false));
    Transform transform = TransformFactory.createTransform(config, "test_replace_all_strings");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();

    assertNotNull(stringFeatures);
    assertEquals(2, stringFeatures.size());

    Set<String> output = stringFeatures.get("bar");

    assertNotNull(output);
    assertEquals(2, output.size());
    assertTrue(output.contains("you like blackberry pie, apple pie; and you also like black!"));
    assertTrue(output.contains("I'm so  excited: you   like black!?!!"));
  }

  @Test
  public void testTransformOverwriteInput() {
    Config config = ConfigFactory.parseString(makeConfig(makeReplacements(), true));
    Transform transform = TransformFactory.createTransform(config, "test_replace_all_strings");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();

    assertNotNull(stringFeatures);
    assertEquals(1, stringFeatures.size());

    Set<String> output = stringFeatures.get("strFeature1");

    assertNotNull(output);
    assertEquals(2, output.size());
    assertTrue(output.contains("you like blackberry pie, apple pie; and you also like black!"));
    assertTrue(output.contains("I'm so  excited: you   like black!?!!"));
  }
}
