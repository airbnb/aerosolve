package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

import static org.junit.Assert.assertTrue;

public class WtaTransformTest {
  private static final Logger log = LoggerFactory.getLogger(WtaTransformTest.class);

  public FeatureVector makeFeatureVector() {
    Map<String, List<Double>> denseFeatures = new HashMap<>();

    List<Double> feature = new ArrayList<>();
    List<Double> feature2 = new ArrayList<>();
    for (int i = 0; i < 100; i++) {
      feature.add(0.1 * i);
      feature2.add(-0.1 * i);
    }
    denseFeatures.put("a", feature);
    denseFeatures.put("b", feature2);
    FeatureVector featureVector = new FeatureVector();
    featureVector.setDenseFeatures(denseFeatures);
    return featureVector;
  }

  public String makeConfig() {
    return "test_wta {\n" +
           " transform : wta\n" +
           " field_names : [ a, b ]\n" +
           " output : wta\n" +
           " seed : 1234\n" +
           " num_words_per_feature : 4\n" +
           " num_tokens_per_word : 4\n"  +
           "}";
  }
  
  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_wta");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);
    assertTrue(featureVector.getStringFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_wta");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    log.info(featureVector.toString());
    assertTrue(featureVector.stringFeatures != null);
    Set<String> wta = featureVector.stringFeatures.get("wta");
    assertTrue(wta != null);
    assertTrue(wta.size() == 8);
    assertTrue(wta.contains("a0:71"));
    assertTrue(wta.contains("a1:60"));
    assertTrue(wta.contains("a2:81"));
    assertTrue(wta.contains("a3:103"));
    assertTrue(wta.contains("b0:34"));
    assertTrue(wta.contains("b1:107"));
    assertTrue(wta.contains("b2:7"));
    assertTrue(wta.contains("b3:193"));
  }
}
