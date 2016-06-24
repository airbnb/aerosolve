package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static junit.framework.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

@Slf4j
public class FloatToDenseTransformTest {
  public String makeConfig() {
    return "test_float_cross_float {\n" +
        " transform : float_to_dense\n" +
        " fields : [floatFeature1,floatFeature1,floatFeature2]\n" +
        " keys : [x,y,z]\n" +
        " string_output : string\n" +
        "}";
  }

  public String notStringConfig() {
    return "test_float_cross_float {\n" +
        " transform : float_to_dense\n" +
        " fields : [floatFeature1,floatFeature1,floatFeature2]\n" +
        " keys : [x,y,z]\n" +
        "}";
  }

  public static FeatureVector makeFeatureVectorFull() {
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Map<String, Double> floatFeature1 = new HashMap<>();

    floatFeature1.put("x", 50.0);
    floatFeature1.put("y", 1.3);
    floatFeature1.put("s", 2000.0);

    Map<String, Double> floatFeature2 = new HashMap<>();

    floatFeature2.put("z", 2000.0);
    floatFeature2.put("k", 2000.0);

    floatFeatures.put("floatFeature1", floatFeature1);
    floatFeatures.put("floatFeature2", floatFeature2);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  public static FeatureVector makeFeatureVectorMissFamily() {
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Map<String, Double> floatFeature1 = new HashMap<>();

    floatFeature1.put("x", 50.0);
    floatFeature1.put("y", 1.3);
    floatFeature1.put("s", 2000.0);

    floatFeatures.put("floatFeature1", floatFeature1);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  public FeatureVector makeFeatureVectorPartial() {
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Map<String, Double> floatFeature1 = new HashMap<>();

    floatFeature1.put("x", 50.0);
    floatFeature1.put("s", 2000.0);

    Map<String, Double> floatFeature2 = new HashMap<>();

    floatFeature2.put("z", 2000.0);
    floatFeature2.put("k", 2000.0);

    floatFeatures.put("floatFeature1", floatFeature1);
    floatFeatures.put("floatFeature2", floatFeature2);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  public FeatureVector makeFeatureVectorFloat() {
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Map<String, Double> floatFeature1 = new HashMap<>();

    floatFeature1.put("x", 50.0);
    floatFeature1.put("s", 2000.0);

    Map<String, Double> floatFeature2 = new HashMap<>();

    floatFeature2.put("k", 2000.0);

    floatFeatures.put("floatFeature1", floatFeature1);
    floatFeatures.put("floatFeature2", floatFeature2);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  public FeatureVector makeFeatureVectorString() {
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Map<String, Double> floatFeature1 = new HashMap<>();
    floatFeature1.put("s", 2000.0);

    Map<String, Double> floatFeature2 = new HashMap<>();

    floatFeature2.put("k", 2000.0);

    floatFeatures.put("floatFeature1", floatFeature1);
    floatFeatures.put("floatFeature2", floatFeature2);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_float_cross_float");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);

    assertTrue(featureVector.getFloatFeatures() == null);
  }

  @Test
  public void testFull() {
    FeatureVector featureVector = testTransform(makeFeatureVectorFull());
    Map<String, List<Double>> denseFeatures = featureVector.getDenseFeatures();

    assertNotNull(denseFeatures);
    assertEquals(1, denseFeatures.size());

    List<Double> out = denseFeatures.get("x^y^z");

    assertEquals(3, out.size());

    assertEquals(50.0, out.get(0), 0.01);
    assertEquals(1.3, out.get(1), 0.01);
    assertEquals(2000, out.get(2), 0.01);
  }

  @Test
  public void testMissFamily() {
    FeatureVector featureVector = testTransform(makeFeatureVectorMissFamily());
    Map<String, List<Double>> denseFeatures = featureVector.getDenseFeatures();

    assertNull(denseFeatures);
  }

  @Test
  public void testPartial() {
    FeatureVector featureVector = testTransform(makeFeatureVectorPartial());
    Map<String, List<Double>> denseFeatures = featureVector.getDenseFeatures();

    assertNull(denseFeatures);
  }

  @Test
  public void testFloat() {
    FeatureVector featureVector = testTransform(makeFeatureVectorFloat());
    Map<String, Map<String, Double>> features = featureVector.getFloatFeatures();

    assertNotNull(features);
    assertEquals(2, features.size());

    Map<String, Double> out = features.get("floatFeature1");

    assertEquals(2, out.size());
  }

  @Test
  public void testNoString() {
    FeatureVector featureVector = testTransform(makeFeatureVectorString(), notStringConfig());
    Map<String, Set<String>> features = featureVector.getStringFeatures();

    assertNull(features);
   }

  public FeatureVector testTransform(FeatureVector featureVector) {
    return testTransform(featureVector, makeConfig());
  }

  public static FeatureVector testTransform(FeatureVector featureVector, String cfg) {
    Config config = ConfigFactory.parseString(cfg);
    Transform transform = TransformFactory.createTransform(config, "test_float_cross_float");
    transform.doTransform(featureVector);
    return featureVector;
  }
}
