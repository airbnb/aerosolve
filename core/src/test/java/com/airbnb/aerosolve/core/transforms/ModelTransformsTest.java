package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.perf.FeatureRegistry;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.airbnb.aerosolve.core.perf.SimpleExample;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

// Tests all the model transforms.
@Slf4j
public class ModelTransformsTest extends BaseTransformTest {

  // Creates a feature vector given a feature family name and latitude and longitude features.
  public MultiFamilyVector makeFeatureVector(
      MultiFamilyVector vector,
      String familyName,
      Double lat,
      Double lng) {
    return TransformTestingHelper.builder(registry, vector)
        .sparse(familyName, "lat", lat)
        .sparse(familyName, "long", lng)
        .dense(familyName + "_dense", new double[]{lat, lng})
        .build();
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
    Example example = new SimpleExample(registry);
    makeFeatureVector(example.context(), "guest_loc", 1.0, 2.0);
    makeFeatureVector(example.createVector(), "host_loc", 3.1, 4.2);
    makeFeatureVector(example.createVector(), "host_loc", 5.3, 6.4);
    return example;
  }

  // Bit of a hack to use the quantize_guest_loc config.  I did this because the main transform
  // won't work with the base test.
  @Override
  public String configKey() {
    return "quantize_guest_loc";
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transformer transformer = new Transformer(config, "model_transforms", registry, null);
    Example example = makeExample();
    example.transform(transformer);

    assertTrue(Iterables.size(example) == 2);
    MultiFamilyVector ex = example.iterator().next();

    assertStringFamily(ex, "guest_loc_quantized", 2,
                       ImmutableSet.of("lat=10", "long=20"));

    assertStringFamily(ex, "host_loc_quantized", 2,
                       ImmutableSet.of("lat=31", "long=42"));

    assertStringFamily(ex, "gxh_loc", 4,
                       ImmutableSet.of("lat=10^lat=31", "long=20^lat=31",
                                       "lat=10^long=42", "long=20^long=42"));

    assertSparseFamily(ex, "guest_loc", 2,
                       ImmutableMap.of("lat", 1.0, "long", 2.0));

    assertSparseFamily(ex, "host_loc", 2,
                       ImmutableMap.of("lat", 3.1, "long", 4.2));

    assertDenseFamily(ex, "guest_loc_dense", new double[]{1.0, 2.0});
    assertDenseFamily(ex, "host_loc_dense", new double[]{3.1, 4.2});

    log.info(example.toString());
  }
}