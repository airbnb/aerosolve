package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * Created by christhetree on 1/27/16.
 */
public class ReplaceAllStringsTransformTest extends BaseTransformTest {

  public String makeConfig() {
    return makeConfig(makeReplacements(), false);
  }

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

  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .string("strFeature1", "I like blueberry pie, apple pie; and I also like blue!")
        .string("strFeature1", "I'm so  excited: I   like blue!?!!")
        .build();
  }

  @Override
  public String configKey() {
    return "test_replace_all_strings";
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
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 2);

    assertStringFamily(featureVector, "bar", 2, ImmutableSet.of(
        "you like blackberry pie, apple pie; and you also like black!",
        "I'm so  excited: you   like black!?!!"
    ));
  }

  @Test
  public void testTransformOverwriteInput() {
    Transform<MultiFamilyVector> transform = getTransform(makeConfig(makeReplacements(), true),
                                                          configKey());
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 1);

    assertStringFamily(featureVector, "strFeature1", 2, ImmutableSet.of(
        "you like blackberry pie, apple pie; and you also like black!",
        "I'm so  excited: you   like black!?!!"
    ));
  }
}
