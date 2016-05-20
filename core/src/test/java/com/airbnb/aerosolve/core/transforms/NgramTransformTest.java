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
 * Created by christhetree on 4/22/16.
 */
public class NgramTransformTest {
  private static final Logger log = LoggerFactory.getLogger(NgramTransformTest.class);

  public String makeConfig(String regex, int n, boolean useMinN) {
    StringBuilder sb = new StringBuilder();
    sb.append("test_ngram {\n");
    sb.append(" transform: ngram\n");
    sb.append(" field1: strFeature1\n");
    sb.append(" regex: ");
    sb.append(regex);
    sb.append("\n");
    sb.append(" n: ");
    sb.append(n);
    sb.append("\n");

    if (useMinN) {
      sb.append(" min_n: 1\n");
    }

    sb.append(" output: bar\n");
    sb.append("}");
    return sb.toString();
  }

  public FeatureVector makeFeatureVector() {
    Map<String, Set<String>> stringFeatures = new HashMap<>();
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Set<String> list = new HashSet<>();
    list.add("I like blueberry pie, apple pie; and I also like blue!");
    list.add("I'm so  excited: I   like blue!?!!");
    stringFeatures.put("strFeature1", list);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig("regex", 2, false));
    Transform transform = TransformFactory.createTransform(config, "test_ngram");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);

    assertTrue(featureVector.getStringFeatures() == null);
    assertTrue(featureVector.getFloatFeatures() == null);
  }

  @Test
  public void testTransform1gram() {
    Config config = ConfigFactory.parseString(makeConfig("\"\"\"[\\s\\p{Punct}]\"\"\"", 1, false));
    Transform transform = TransformFactory.createTransform(config, "test_ngram");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();

    assertEquals(1, stringFeatures.size());
    assertEquals(1, floatFeatures.size());

    Map<String, Double> output = floatFeatures.get("bar");

    assertEquals(11, output.size());
    assertEquals(1.0, output.get("apple"), 0.0);
    assertEquals(1.0, output.get("blueberry"), 0.0);
    assertEquals(2.0, output.get("blue"), 0.0);
    assertEquals(3.0, output.get("like"), 0.0);
    assertEquals(1.0, output.get("excited"), 0.0);
    assertEquals(1.0, output.get("and"), 0.0);
    assertEquals(4.0, output.get("I"), 0.0);
    assertEquals(1.0, output.get("also"), 0.0);
    assertEquals(1.0, output.get("so"), 0.0);
    assertEquals(2.0, output.get("pie"), 0.0);
    assertEquals(1.0, output.get("m"), 0.0);
  }

  @Test
  public void testTransform2gram() {
    Config config = ConfigFactory.parseString(makeConfig("\"\"\"[\\s\\p{Punct}]\"\"\"", 2, false));
    Transform transform = TransformFactory.createTransform(config, "test_ngram");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();

    assertEquals(1, stringFeatures.size());
    assertEquals(1, floatFeatures.size());

    Map<String, Double> output = floatFeatures.get("bar");

    assertEquals(14, output.size());
    assertEquals(2.0, output.get("I" + NgramTransform.BIGRAM_SEPARATOR + "like"), 0.0);
    assertEquals(1.0, output.get("like" + NgramTransform.BIGRAM_SEPARATOR + "blueberry"), 0.0);
    assertEquals(1.0, output.get("blueberry" + NgramTransform.BIGRAM_SEPARATOR + "pie"), 0.0);
    assertEquals(1.0, output.get("pie" + NgramTransform.BIGRAM_SEPARATOR + "apple"), 0.0);
    assertEquals(1.0, output.get("apple" + NgramTransform.BIGRAM_SEPARATOR + "pie"), 0.0);
    assertEquals(1.0, output.get("pie" + NgramTransform.BIGRAM_SEPARATOR + "and"), 0.0);
    assertEquals(1.0, output.get("and" + NgramTransform.BIGRAM_SEPARATOR + "I"), 0.0);
    assertEquals(1.0, output.get("I" + NgramTransform.BIGRAM_SEPARATOR + "also"), 0.0);
    assertEquals(1.0, output.get("also" + NgramTransform.BIGRAM_SEPARATOR + "like"), 0.0);
    assertEquals(2.0, output.get("like" + NgramTransform.BIGRAM_SEPARATOR + "blue"), 0.0);
    assertEquals(1.0, output.get("I" + NgramTransform.BIGRAM_SEPARATOR + "m"), 0.0);
    assertEquals(1.0, output.get("m" + NgramTransform.BIGRAM_SEPARATOR + "so"), 0.0);
    assertEquals(1.0, output.get("so" + NgramTransform.BIGRAM_SEPARATOR + "excited"), 0.0);
    assertEquals(1.0, output.get("excited" + NgramTransform.BIGRAM_SEPARATOR + "I"), 0.0);
  }

  @Test
  public void testTransform11gram() {
    Config config = ConfigFactory.parseString(makeConfig("\"\"\"[\\s\\p{Punct}]\"\"\"", 11, false));
    Transform transform = TransformFactory.createTransform(config, "test_ngram");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();

    assertEquals(1, stringFeatures.size());
    assertEquals(1, floatFeatures.size());

    Map<String, Double> output = floatFeatures.get("bar");

    assertEquals(1, output.size());
  }

  @Test
  public void testTransform99gram() {
    Config config = ConfigFactory.parseString(makeConfig("\"\"\"[\\s\\p{Punct}]\"\"\"", 99, false));
    Transform transform = TransformFactory.createTransform(config, "test_ngram");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();

    assertEquals(1, stringFeatures.size());
    assertEquals(1, floatFeatures.size());

    Map<String, Double> output = floatFeatures.get("bar");

    assertEquals(0, output.size());
  }

  @Test
  public void testTransformUseMinN() {
    Config config = ConfigFactory.parseString(makeConfig("\"\"\"[\\s\\p{Punct}]\"\"\"", 2, true));
    Transform transform = TransformFactory.createTransform(config, "test_ngram");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();

    assertEquals(1, stringFeatures.size());
    assertEquals(1, floatFeatures.size());

    Map<String, Double> output = floatFeatures.get("bar");

    assertEquals(25, output.size());
    assertEquals(1.0, output.get("apple"), 0.0);
    assertEquals(1.0, output.get("blueberry"), 0.0);
    assertEquals(2.0, output.get("blue"), 0.0);
    assertEquals(3.0, output.get("like"), 0.0);
    assertEquals(1.0, output.get("excited"), 0.0);
    assertEquals(1.0, output.get("and"), 0.0);
    assertEquals(4.0, output.get("I"), 0.0);
    assertEquals(1.0, output.get("also"), 0.0);
    assertEquals(1.0, output.get("so"), 0.0);
    assertEquals(2.0, output.get("pie"), 0.0);
    assertEquals(1.0, output.get("m"), 0.0);
    assertEquals(2.0, output.get("I" + NgramTransform.BIGRAM_SEPARATOR + "like"), 0.0);
    assertEquals(1.0, output.get("like" + NgramTransform.BIGRAM_SEPARATOR + "blueberry"), 0.0);
    assertEquals(1.0, output.get("blueberry" + NgramTransform.BIGRAM_SEPARATOR + "pie"), 0.0);
    assertEquals(1.0, output.get("pie" + NgramTransform.BIGRAM_SEPARATOR + "apple"), 0.0);
    assertEquals(1.0, output.get("apple" + NgramTransform.BIGRAM_SEPARATOR + "pie"), 0.0);
    assertEquals(1.0, output.get("pie" + NgramTransform.BIGRAM_SEPARATOR + "and"), 0.0);
    assertEquals(1.0, output.get("and" + NgramTransform.BIGRAM_SEPARATOR + "I"), 0.0);
    assertEquals(1.0, output.get("I" + NgramTransform.BIGRAM_SEPARATOR + "also"), 0.0);
    assertEquals(1.0, output.get("also" + NgramTransform.BIGRAM_SEPARATOR + "like"), 0.0);
    assertEquals(2.0, output.get("like" + NgramTransform.BIGRAM_SEPARATOR + "blue"), 0.0);
    assertEquals(1.0, output.get("I" + NgramTransform.BIGRAM_SEPARATOR + "m"), 0.0);
    assertEquals(1.0, output.get("m" + NgramTransform.BIGRAM_SEPARATOR + "so"), 0.0);
    assertEquals(1.0, output.get("so" + NgramTransform.BIGRAM_SEPARATOR + "excited"), 0.0);
    assertEquals(1.0, output.get("excited" + NgramTransform.BIGRAM_SEPARATOR + "I"), 0.0);
  }
}
