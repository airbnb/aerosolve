package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

@Slf4j
public class WtaTransformTest extends BaseTransformTest {

  public MultiFamilyVector makeFeatureVector() {
    int max = 100;
    double[] feature = new double[max];
    double[] feature2 = new double[max];
    for (int i = 0; i < max; i++) {
      feature[i] = 0.1 * i;
      feature2[i] = -0.1 * i;
    }

    return TransformTestingHelper.builder(registry)
        .dense("a", feature)
        .dense("b", feature2)
        .build();
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

  @Override
  public String configKey() {
    return "test_wta";
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    log.info(featureVector.toString());

    assertTrue(featureVector.numFamilies() == 3);

    // TODO (Brad): Because of the change to not re-seed the random on each family, this now
    // produces different result for the second family ("b").  Maybe revisit but this seems
    // reasonable to change the values.
    assertStringFamily(featureVector, "wta", 8, ImmutableSet.of(
        "a0:71", "a1:60", "a2:81", "a3:103", "b0:254", "b1:104", "b2:134", "b3:21"
    ));
  }
}
