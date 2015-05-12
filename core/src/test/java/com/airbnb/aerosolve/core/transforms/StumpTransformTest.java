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
public class StumpTransformTest {
  private static final Logger log = LoggerFactory.getLogger(StumpTransformTest.class);

  public FeatureVector makeFeatureVector() {
    Map<String, Set<String>> stringFeatures = new HashMap<>();
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Set list = new HashSet<String>();
    list.add("aaa");
    list.add("bbb");
    stringFeatures.put("strFeature1", list);

    Map<String, Double> map = new HashMap<>();
    map.put("lat", 37.7);
    map.put("long", 40.0);
    floatFeatures.put("loc", map);

    Map<String, Double> map2 = new HashMap<>();
    map2.put("foo", 1.0);
    floatFeatures.put("F", map2);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    featureVector.setFloatFeatures(floatFeatures);
    return featureVector;
  }

  public String makeConfig() {
    return "test_stump {\n" +
           " transform : stump\n" +
           " stumps : [\n" +
           " \"loc,lat,30.0,lat>=30.0\"\n"+
           " \"loc,lng,50.0,lng>=50.0\"\n"+
           " \"fake,fake,50.0,fake>50.0\"\n"+
           " \"F,foo,0.0,foo>=0.0\"\n"+
           " ]\n" +
           " output : bar\n" +
           "}";
  }
  
  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_stump");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);
    assertTrue(featureVector.getStringFeatures() == null);
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_stump");
    FeatureVector featureVector = makeFeatureVector();
    transform.doTransform(featureVector);
    Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    assertTrue(stringFeatures.size() == 2);

    Set<String> out = featureVector.stringFeatures.get("bar");
    for (String entry : out) {
      log.info(entry);
    }
    assertTrue(out.contains("lat>=30.0"));
    assertTrue(out.contains("foo>=0.0"));
  }
}