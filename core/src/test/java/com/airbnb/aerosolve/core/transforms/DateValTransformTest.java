package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.perf.MultiFamilyVector;
import com.google.common.collect.ImmutableMap;
import org.junit.Test;

import static org.junit.Assert.assertTrue;


/**
 * Created by seckcoder on 12/20/15.
 */
public class DateValTransformTest extends BaseTransformTest {
  public String makeConfig() {
    return makeConfig("day_of_month");
  }
  public String makeConfig(String dateType) {
    return "test_date {\n" +
            " transform: date_val\n" +
            " field1: dates\n" +
            " date_type: " + dateType + "\n" +
            " output: bar\n" +
            "}";
  }
  public MultiFamilyVector makeFeatureVector() {
    return TransformTestingHelper.builder(registry)
        .string("dates", "2009-03-01")
        .string("dates", "2009-02-27")
        .build();
  }

  @Override
  public String configKey() {
    return "test_date";
  }

  @Test
  public void testDayOfMonthTransform() {
    Transform<MultiFamilyVector> transform = getTransform(makeConfig("day_of_month"), configKey());
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);
    assertTrue(featureVector.numFamilies() == 2);

    assertSparseFamily(featureVector, "bar", 2, ImmutableMap.of(
        "2009-03-01", 1.0,
        "2009-02-27", 27.0
    ));
  }

  @Test
  public void testDayOfWeekTransform() {
    Transform<MultiFamilyVector> transform = getTransform(makeConfig("day_of_week"), configKey());
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);
    assertTrue(featureVector.numFamilies() == 2);

    assertSparseFamily(featureVector, "bar", 2, ImmutableMap.of(
        "2009-03-01", 1.0,
        "2009-02-27", 6.0
    ));
  }

  @Test
  public void testDayOfYearTransform() {
    Transform<MultiFamilyVector> transform = getTransform(makeConfig("day_of_year"), configKey());
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);
    assertTrue(featureVector.numFamilies() == 2);

    assertSparseFamily(featureVector, "bar", 2, ImmutableMap.of(
        "2009-03-01", 60.0,
        "2009-02-27", 58.0
    ));
  }

  @Test
  public void testYearOfDateTransform() {
    Transform<MultiFamilyVector> transform = getTransform(makeConfig("year"), configKey());
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);
    assertTrue(featureVector.numFamilies() == 2);

    assertSparseFamily(featureVector, "bar", 2, ImmutableMap.of(
        "2009-03-01", 2009.0,
        "2009-02-27", 2009.0
    ));
  }

  @Test
  public void testMonthOfDateTransform() {
    Transform<MultiFamilyVector> transform = getTransform(makeConfig("month"), configKey());
    MultiFamilyVector featureVector = makeFeatureVector();
    transform.apply(featureVector);
    assertTrue(featureVector.numFamilies() == 2);

    assertSparseFamily(featureVector, "bar", 2, ImmutableMap.of(
        "2009-03-01", 3.0,
        "2009-02-27", 2.0
    ));
  }
}
