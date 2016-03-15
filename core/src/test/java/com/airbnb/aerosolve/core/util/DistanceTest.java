package com.airbnb.aerosolve.core.util;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class DistanceTest {

  @Test
  public void miles() throws Exception {
    assertEquals(262.6777938054349, Distance.miles(32.9697, -96.80322, 29.46786, -98.53506), 0.01);
  }

  @Test
  public void kilometers() throws Exception {
    assertEquals(422.73893139401383, Distance.kilometers(32.9697, -96.80322, 29.46786, -98.53506), 0.01);
  }

  @Test
  public void nauticalMiles() throws Exception {
    assertEquals(228.10939614063963, Distance.nauticalMiles(32.9697, -96.80322, 29.46786, -98.53506), 0.01);
  }
}