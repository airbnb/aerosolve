package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

import static org.junit.Assert.assertTrue;

// Tests all the model transforms.
public class ModelTransformsTest {
  private static final Logger log = LoggerFactory.getLogger(ModelTransformsTest.class);

  // Creates a feature vector given a feature family name and latitude and longitude features.
  public FeatureVector makeFeatureVector(String familyName,
                                         Double lat,
                                         Double lng) {
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();
    Map<String, List<Double>> denseFeatures = new HashMap<>();

    Map<String, Double> map = new HashMap<>();
    map.put("lat", lat);
    map.put("long", lng);
    floatFeatures.put(familyName, map);

    List<Double> list = new ArrayList<>();
    list.add(lat);
    list.add(lng);
    String denseFamilyName = familyName + "_dense";
    denseFeatures.put(denseFamilyName, list);

    FeatureVector featureVector = new FeatureVector();
    featureVector.setFloatFeatures(floatFeatures);
    featureVector.setDenseFeatures(denseFeatures);
    return featureVector;
  }

  public String makeConfig() {
    return "quantize_guest_loc {\n" +
           " transform : quantize\n" +
           " field1 : guest_loc\n" +
           " scale : 10\n" +
           " output : guest_loc_quantized\n" +
           "}\n" +
           "quantize_host_loc {\n" +
           " transform : quantize\n" +
           " field1 : host_loc\n" +
           " scale : 10\n" +
           " output : host_loc_quantized\n" +
           "}\n" +
           "cross_guest_host_loc {\n" +
           " transform : cross\n" +
           " field1 : guest_loc_quantized\n" +
           " field2 : host_loc_quantized\n" +
           " output : gxh_loc\n" +
           "}\n" +
           "context_transform {\n" +
           " transform : list\n" +
           " transforms : [quantize_guest_loc]\n" +
           "}\n" +
           "item_transform {\n" +
           " transform : list\n" +
           " transforms : [quantize_host_loc]\n" +
           "}\n" +
           "combined_transform {\n" +
           " transform : list\n" +
           " transforms : [cross_guest_host_loc]\n" +
           "}\n" +
           "model_transforms {\n" +
           " context_transform : context_transform\n" +
           " item_transform : item_transform\n" +
           " combined_transform : combined_transform\n" +
           "}";
  }

  private Example makeExample() {
    Example example = new Example();
    example.setContext(makeFeatureVector("guest_loc", 1.0, 2.0));
    example.addToExample(makeFeatureVector("host_loc", 3.1, 4.2));
    example.addToExample(makeFeatureVector("host_loc", 5.3, 6.4));
    return example;
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transformer transformer = new Transformer(config, "model_transforms");
    Example example = makeExample();
    transformer.combineContextAndItems(example);
    assertTrue(example.example.size() == 2);
    FeatureVector ex = example.example.get(0);
    assertTrue(ex.stringFeatures.size() == 3);
    assertTrue(ex.stringFeatures.get("guest_loc_quantized").contains("lat=10"));
    assertTrue(ex.stringFeatures.get("guest_loc_quantized").contains("long=20"));
    assertTrue(ex.stringFeatures.get("host_loc_quantized").contains("lat=31"));
    assertTrue(ex.stringFeatures.get("host_loc_quantized").contains("long=42"));
    assertTrue(ex.stringFeatures.get("gxh_loc").contains("lat=10^lat=31"));
    assertTrue(ex.stringFeatures.get("gxh_loc").contains("long=20^lat=31"));
    assertTrue(ex.stringFeatures.get("gxh_loc").contains("lat=10^long=42"));
    assertTrue(ex.stringFeatures.get("gxh_loc").contains("long=20^long=42"));
    assertTrue(ex.floatFeatures.get("guest_loc").get("lat") == 1.0);
    assertTrue(ex.floatFeatures.get("guest_loc").get("long") == 2.0);
    assertTrue(ex.floatFeatures.get("host_loc").get("lat") == 3.1);
    assertTrue(ex.floatFeatures.get("host_loc").get("long") == 4.2);
    assertTrue(ex.denseFeatures.get("guest_loc_dense").contains(1.0));
    assertTrue(ex.denseFeatures.get("guest_loc_dense").contains(2.0));
    assertTrue(ex.denseFeatures.get("host_loc_dense").contains(3.1));
    assertTrue(ex.denseFeatures.get("host_loc_dense").contains(4.2));
    log.info(example.toString());
  }
}