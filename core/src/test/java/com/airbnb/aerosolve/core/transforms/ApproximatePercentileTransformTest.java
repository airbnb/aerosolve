package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class ApproximatePercentileTransformTest {
  private static final Logger log = LoggerFactory.getLogger(ApproximatePercentileTransformTest.class);

  public FeatureVector makeFeatureVector(double low, double high, double val) {
    Map<String, Set<String>> stringFeatures = new HashMap<>();
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Set list = new HashSet<String>();
    list.add("aaa");
    list.add("bbb");
    stringFeatures.put("strFeature1", list);

    Map<String, Double> map = new HashMap<>();
    map.put("10th", low);
    map.put("90th", high);
    floatFeatures.put("DECILES", map);

    Map<String, Double> map2 = new HashMap<>();
    map2.put("foo", val);
    floatFeatures.put("F", map2);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  public String makeConfig() {
    return "test_approximate_percentile {\n" +
           " transform : approximate_percentile\n" +
           " field1 : DECILES\n" +
           " low : 10th\n" +
           " upper : 90th\n" +
           " minDiff : 10 \n" +
           " field2 : F\n" +
           " key2 : foo\n" +
           " output : PERCENTILE\n" +
           " outputKey : percentile\n" +
           "}";
  }
  
  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_approximate_percentile");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);
    assertTrue(featureVector.getStringFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_approximate_percentile");

    double[] values = { -1.0, 10.0, 15.0, 20.0, 50.0, 60.0, 100.0, 200.0 };
    double[] expected = { 0.0, 0.0, 0.05, 0.11, 0.44, 0.55, 1.0, 1.0 };

    for (int i = 0; i < values.length; i++) {
      double val = values[i];

      FeatureVector featureVector = makeFeatureVector(10.0, 100.0, val);
      transform.doTransform(featureVector);
      Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
      assertTrue(stringFeatures.size() == 1);

      Map<String, Double> out = featureVector.floatFeatures.get("PERCENTILE");
      assertTrue(out.size() == 1);
      assertEquals(expected[i], out.get("percentile"), 0.01);
    }
  }

  @Test
  public void testAbstain() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_approximate_percentile");

    FeatureVector featureVector = makeFeatureVector(10.0, 11.0, 1.0);
    transform.doTransform(featureVector);
    assertTrue(featureVector.floatFeatures.get("PERCENTILE") == null);
  }
}