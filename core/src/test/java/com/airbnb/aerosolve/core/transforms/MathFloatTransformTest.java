package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;

import java.util.Map;
import java.util.Set;

import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class MathFloatTransformTest {
  private static final Logger log = LoggerFactory.getLogger(MathFloatTransformTest.class);

  public String makeConfig(String functionName) {
    return "test_math {\n" +
        " transform : math_float\n" +
        " field1 : loc\n" +
        " keys : [lat,long,z]\n" +
        " output : bar\n" +
        " function : " + functionName +
        "}";
  }

  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig("log10"));
    Transform transform = TransformFactory.createTransform(config, "test_math");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);
    assertTrue(featureVector.getFloatFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig("log10"));
    Transform transform = TransformFactory.createTransform(config, "test_math");
    FeatureVector featureVector = TransformTestingHelper.makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 1);

    Map<String, Double> feat1 = featureVector.getFloatFeatures().get("loc");
    // the original features are not changed
    assertEquals(3, feat1.size());
    assertEquals(37.7, feat1.get("lat"), 0.1);
    assertEquals(40.0, feat1.get("long"), 0.1);
    assertEquals(-20.0, feat1.get("z"), 0.1);

    Map<String, Double> feat2 = featureVector.getFloatFeatures().get("bar");
    assertEquals(3, feat2.size());
    assertEquals(Math.log10(37.7), feat2.get("lat"), 0.1);
    assertEquals(Math.log10(40.0), feat2.get("long"), 0.1);
    // for negative value, it would be a missing feature
    assertTrue(!feat2.containsKey("z"));
    // existing feature in 'bar' should not change
    assertEquals(1.0, feat2.get("bar_fv"), 0.1);

    // test an undefined function
    Config config2 = ConfigFactory.parseString(makeConfig("tan"));
    Transform transform2 = TransformFactory.createTransform(config2, "test_math");
    FeatureVector featureVector2 = TransformTestingHelper.makeFeatureVector();
    transform2.doTransform(featureVector2);
    // the original features are unchanged
    Map<String, Double> feat3 = featureVector2.getFloatFeatures().get("loc");
    assertEquals(3, feat3.size());
    assertEquals(37.7, feat3.get("lat"), 0.1);
    assertEquals(40.0, feat3.get("long"), 0.1);
    assertEquals(-20.0, feat3.get("z"), 0.1);
    // new features should not exist
    Map<String, Double> feat4 = featureVector2.getFloatFeatures().get("bar");
    assertEquals(1, feat4.size());
    assertEquals(1.0, feat4.get("bar_fv"), 0.1);
  }
}
