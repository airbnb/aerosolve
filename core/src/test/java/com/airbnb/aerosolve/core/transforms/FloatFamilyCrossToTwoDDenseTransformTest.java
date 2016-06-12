package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.FeatureVector;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import java.util.List;
import java.util.Map;

import static com.airbnb.aerosolve.core.transforms.FloatToDenseTransformTest.makeFeatureVectorFull;
import static com.airbnb.aerosolve.core.transforms.FloatToDenseTransformTest.makeFeatureVectorMissFamily;
import static junit.framework.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

@Slf4j
public class FloatFamilyCrossToTwoDDenseTransformTest {
  public String makeConfig() {
    return "test_float_cross_float {\n" +
        " transform : float_family_cross_to_two_d_dense\n" +
        " field1 : floatFeature1 \n" +
        " field2 : floatFeature2 \n" +
        "}";
  }

  public String makeSelfCrossConfig() {
    return "test_float_cross_float {\n" +
        " transform : float_family_cross_to_two_d_dense\n" +
        " field1 : floatFeature1 \n" +
        "}";
  }

  @Test
  public void testEmptyFeatureVector() {
    Config config = ConfigFactory.parseString(makeConfig());
    Transform transform = TransformFactory.createTransform(config, "test_float_cross_float");
    FeatureVector featureVector = new FeatureVector();
    transform.doTransform(featureVector);

    assertNull(featureVector.getFloatFeatures());
  }

  public FeatureVector testTransform(FeatureVector featureVector) {
    return FloatToDenseTransformTest.testTransform(featureVector, makeConfig());
  }

  @Test
  public void testFull() {
    FeatureVector featureVector = testTransform(makeFeatureVectorFull());
    Map<String, List<Double>> denseFeatures = featureVector.getDenseFeatures();

    assertNotNull(denseFeatures);
    assertEquals(6, denseFeatures.size());

    List<Double> out = denseFeatures.get("x^z");

    assertEquals(2, out.size());

    assertEquals(50.0, out.get(0), 0.01);
    assertEquals(2000, out.get(1), 0.01);
    assertNull(denseFeatures.get("x^y"));
    assertNotNull(denseFeatures.get("s^k"));
  }

  @Test
  public void testMissFamily() {
    FeatureVector featureVector = testTransform(makeFeatureVectorMissFamily());
    Map<String, List<Double>> denseFeatures = featureVector.getDenseFeatures();

    assertNull(denseFeatures);
  }

  @Test
  public void testSelfCross() {
    FeatureVector featureVector = FloatToDenseTransformTest.testTransform(
        makeFeatureVectorFull(), makeSelfCrossConfig());
    Map<String, List<Double>> denseFeatures = featureVector.getDenseFeatures();

    assertNotNull(denseFeatures);
    assertEquals(3, denseFeatures.size());
    log.debug("dense {}", denseFeatures);
    List<Double> out = denseFeatures.get("s^x");

    assertEquals(2, out.size());

    assertEquals(2000, out.get(0), 0.01);
    assertEquals(50.0, out.get(1), 0.01);
    assertNull(denseFeatures.get("x^x"));
    assertNotNull(denseFeatures.get("s^y"));
    assertNotNull(denseFeatures.get("x^y"));
  }

}
