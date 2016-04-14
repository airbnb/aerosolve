package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
@Slf4j
public class KdtreeTransformTest extends BaseTransformTest {

  public String makeConfig() {
    return "test_kdtree {\n" +
           " transform : kdtree\n" +
           " include \"test_kdt.model.conf\"\n" +
           " field1 : loc\n" +
           " value1 : lat\n" +
           " value2 : long\n" +
           " max_count : 2\n" +
           " output : loc_kdt\n" +
           "}";
  }

  @Override
  public String configKey() {
    return "test_kdtree";
  }

  @Test
  public void testTransform() {
    Config config = ConfigFactory.parseString(makeConfig());
    log.info("Model encoded is " + config.getString("test_kdtree.model_base64"));
    Transform<MultiFamilyVector> transform =
        TransformFactory.createTransform(config, "test_kdtree", registry, null);
    MultiFamilyVector featureVector = TransformTestingHelper.makeSimpleVector(registry);
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 3);

    assertStringFamily(featureVector, "loc_kdt", 2, ImmutableSet.of("2", "4"));
  }
}