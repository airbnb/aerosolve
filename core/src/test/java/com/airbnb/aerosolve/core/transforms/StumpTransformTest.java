package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class StumpTransformTest extends BaseTransformTest {

  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .simpleStrings()
        .location()
        .sparse("F", "foo", 1.0)
        .build();
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

  @Override
  public String configKey() {
    return "test_stump";
  }

  @Test
  public void testTransform() {
    Transform<MultiFamilyVector> transform = getTransform();
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);

    assertTrue(featureVector.numFamilies() == 4);

    assertStringFamily(featureVector, "bar", 2, ImmutableSet.of(
        "lat>=30.0", "foo>=0.0"
    ));
  }
}