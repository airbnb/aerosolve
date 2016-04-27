package com.airbnb.aerosolve.core.features;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class FeatureMappingTest {

  @Test
  public void add() throws Exception {
    FeatureMapping m = new FeatureMapping(100);
    String[] doubleNames = {"a", "b"};
    m.add(Double.class, doubleNames);
    String[] booleanNames = {"c", "d"};
    m.add(Boolean.class, booleanNames);
    String[] strNames = {"e", "f"};
    m.add(String.class, strNames);
    m.finish();

    assertEquals(m.getNames().length, 6);
    assertArrayEquals(m.getNames(),
        new String[]{"a", "b", "c", "d", "e", "f"});
    assertEquals(m.getMapping().get(String.class).start, 4);
    assertEquals(m.getMapping().get(String.class).length, 2);
  }
}