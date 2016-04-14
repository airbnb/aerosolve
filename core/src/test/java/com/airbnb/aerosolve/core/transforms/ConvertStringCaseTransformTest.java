package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import java.util.Set;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * Created by christhetree on 1/27/16.
 */
public class ConvertStringCaseTransformTest extends BaseTransformTest {

  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .string("strFeature1", "I like BLUEBERRY pie, APPLE pie; and I also like BLUE!")
        .string("strFeature1", "I'm so  excited: I   like blue!?!!")
        .build();
  }

  public String makeConfig(boolean convertToUppercase, boolean overwriteInput) {
    StringBuilder sb = new StringBuilder();
    sb.append("test_convert_string_case {\n");
    sb.append(" transform: convert_string_case\n");
    sb.append(" field1: strFeature1\n");
    sb.append(" convert_to_uppercase: ");
    sb.append(Boolean.toString(convertToUppercase));
    sb.append("\n");

    if (!overwriteInput) {
      sb.append(" output: bar\n");
    }

    sb.append("}");

    return sb.toString();
  }

  public String makeConfig() {
    return makeConfig(false, false);
  }

  @Override
  public String configKey() {
    return "test_convert_string_case";
  }

  @Test
  public void testTransformConvertToLowercase() {
    doTest(false, false, ImmutableSet.of(
        "i like blueberry pie, apple pie; and i also like blue!",
        "i'm so  excited: i   like blue!?!!"),
         2, "bar");
  }

  @Test
  public void testTransformConvertToUppercase() {
    doTest(true, false, ImmutableSet.of(
        "I LIKE BLUEBERRY PIE, APPLE PIE; AND I ALSO LIKE BLUE!",
        "I'M SO  EXCITED: I   LIKE BLUE!?!!"),
         2, "bar");
  }

  private void doTest(boolean convertToUpperCase, boolean overwriteInput,
                      Set<String> expected, int numFamilies, String family) {
    Transform<MultiFamilyVector> transform = getTransform(
        makeConfig(convertToUpperCase, overwriteInput), configKey());
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == numFamilies);

    assertStringFamily(featureVector, family, 2, expected);
  }

  @Test
  public void testTransformOverwriteInput() {
    doTest(true, true, ImmutableSet.of(
        "I LIKE BLUEBERRY PIE, APPLE PIE; AND I ALSO LIKE BLUE!",
        "I'M SO  EXCITED: I   LIKE BLUE!?!!"),
        1, "strFeature1");
  }
}
