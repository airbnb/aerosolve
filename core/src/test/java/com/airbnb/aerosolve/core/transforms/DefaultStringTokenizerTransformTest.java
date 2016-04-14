package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableMap;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * Created by christhetree on 1/27/16.
 */
public class DefaultStringTokenizerTransformTest extends BaseTransformTest {

  public String makeConfig() {
    return makeConfig("regex", false);
  }

  public String makeConfig(String regex, boolean generateBigrams) {
    return "test_tokenizer {\n" +
        " transform: default_string_tokenizer\n" +
        " field1: strFeature1\n" +
        " regex: " + regex + "\n" +
        " output: bar\n" +
        " generate_bigrams: " + Boolean.toString(generateBigrams) + "\n" +
        " bigrams_output: bigrams\n" +
        "}";
  }

  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .string("strFeature1", "I like blueberry pie, apple pie; and I also like blue!")
        .string("strFeature1", "I'm so  excited: I   like blue!?!!")
        .build();
  }

  @Override
  public String configKey() {
    return "test_tokenizer";
  }

  @Test
  public void testTransformWithoutBigrams() {
    Transform<MultiFamilyVector> transform = getTransform(makeConfig("\"\"\"[\\s\\p{Punct}]\"\"\"", false),
                                       configKey());
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 2);

    assertSparseFamily(featureVector, "bar", 11, ImmutableMap.<String, Double>builder()
        .put("apple", 1.0)
        .put("blueberry", 1.0)
        .put("blue", 2.0)
        .put("like", 3.0)
        .put("excited", 1.0)
        .put("and", 1.0)
        .put("I", 4.0)
        .put("also", 1.0)
        .put("so", 1.0)
        .put("pie", 2.0)
        .put("m", 1.0)
        .build());
  }

  @Test
  public void testTransformWithBigrams() {
    Transform<MultiFamilyVector> transform = getTransform(makeConfig("\"\"\"[\\s\\p{Punct}]\"\"\"", true),
                                       configKey());
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 3);

    assertSparseFamily(featureVector, "bar", 11, ImmutableMap.<String, Double>builder()
        .put("apple", 1.0)
        .put("blueberry", 1.0)
        .put("blue", 2.0)
        .put("like", 3.0)
        .put("excited", 1.0)
        .put("and", 1.0)
        .put("I", 4.0)
        .put("also", 1.0)
        .put("so", 1.0)
        .put("pie", 2.0)
        .put("m", 1.0)
        .build());

    assertSparseFamily(featureVector, "bigrams", 14, ImmutableMap.<String, Double>builder()
        .put("I" + DefaultStringTokenizerTransform.BIGRAM_SEPARATOR + "like", 2.0)
        .put("like" + DefaultStringTokenizerTransform.BIGRAM_SEPARATOR + "blueberry", 1.0)
        .put("blueberry" + DefaultStringTokenizerTransform.BIGRAM_SEPARATOR + "pie", 1.0)
        .put("pie" + DefaultStringTokenizerTransform.BIGRAM_SEPARATOR + "apple", 1.0)
        .put("apple" + DefaultStringTokenizerTransform.BIGRAM_SEPARATOR + "pie", 1.0)
        .put("pie" + DefaultStringTokenizerTransform.BIGRAM_SEPARATOR + "and", 1.0)
        .put("and" + DefaultStringTokenizerTransform.BIGRAM_SEPARATOR + "I", 1.0)
        .put("I" + DefaultStringTokenizerTransform.BIGRAM_SEPARATOR + "also", 1.0)
        .put("also" + DefaultStringTokenizerTransform.BIGRAM_SEPARATOR + "like", 1.0)
        .put("like" + DefaultStringTokenizerTransform.BIGRAM_SEPARATOR + "blue", 2.0)
        .put("I" + DefaultStringTokenizerTransform.BIGRAM_SEPARATOR + "m", 1.0)
        .put("m" + DefaultStringTokenizerTransform.BIGRAM_SEPARATOR + "so", 1.0)
        .put("so" + DefaultStringTokenizerTransform.BIGRAM_SEPARATOR + "excited", 1.0)
        .put("excited" + DefaultStringTokenizerTransform.BIGRAM_SEPARATOR + "I", 1.0)
        .build());
  }
}
