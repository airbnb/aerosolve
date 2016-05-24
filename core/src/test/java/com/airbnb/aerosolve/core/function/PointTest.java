package com.airbnb.aerosolve.core.function;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class PointTest {

  private Point example() {
    return new Point(-6.0f);
  }

  @Test
  public void LInfinityCap() throws Exception {
    Point point = example();
    point.LInfinityCap(3.0f);
    assertEquals(-3, point.evaluate(1.0f), 0);
  }

}
