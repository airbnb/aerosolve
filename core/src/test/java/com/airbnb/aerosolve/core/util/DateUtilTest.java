package com.airbnb.aerosolve.core.util;

import org.junit.Test;

import java.util.Map;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class DateUtilTest {

  @Test
  public void applyForDateRange() throws Exception {
    Function<String, String> f = d -> {
      return d;
    };
    Map<String, String> map = DateUtil.applyForDateRange("2016-02-23", "2016-03-23", f);

    assertEquals(29, map.size());
    assertEquals("2016-02-29", map.get("2016-02-29"));
  }
}