package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableMap;
import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class ApproximatePercentileTransformTest extends BaseTransformTest {

  public MultiFamilyVector makeFeatureVector(double low, double high, double val) {
    return TransformTestingHelper.builder(registry)
        .simpleStrings()
        .sparse("DECILES", "10th", low)
        .sparse("DECILES", "90th", high)
        .sparse("F", "foo", val)
        .build();
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

  @Override
  public String configKey() {
    return "test_approximate_percentile";
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();

    double[] values = { -1.0, 10.0, 15.0, 20.0, 50.0, 60.0, 100.0, 200.0 };
    double[] expected = { 0.0, 0.0, 0.05, 0.11, 0.44, 0.55, 1.0, 1.0 };

    for (int i = 0; i < values.length; i++) {
      double val = values[i];

      MultiFamilyVector featureVector = makeFeatureVector(10.0, 100.0, val);
      transform.apply(featureVector);
      assertTrue(featureVector.numFamilies() == 4);

      assertSparseFamily(featureVector, "PERCENTILE", 1,
                         ImmutableMap.of("percentile", expected[i]));
    }
  }

  @Test
  public void testAbstain() {
    Transform<MultiFamilyVector> transform = getTransform();

    MultiFamilyVector featureVector = makeFeatureVector(10.0, 11.0, 1.0);
    transform.apply(featureVector);
    assertFalse(featureVector.contains(registry.family("PERCENTILE")));
  }
}