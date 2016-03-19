package com.airbnb.aerosolve.core.features;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class FeatureMappingTest {

  @Test
  public void add() throws Exception {
    FeatureMapping m = new FeatureMapping(100);
    String[] doubleNames = {"a", "b"};
    m.add(Double.class, doubleNames, 2);
    String[] booleanNames = {"c", "d"};
    m.add(Boolean.class, booleanNames, 3);
    String[] strNames = {"e", "f"};
    m.add(String.class, strNames, 1);
    m.finish();

    assertEquals(m.getNames().length, 6);
    assertArrayEquals(m.getNames(),
        new String[]{"a", "b", "c", "d", "e", "f"});
    assertArrayEquals(m.getTypes(),
        new Integer[]{2, 2, 3, 3, 1, 1});
    assertEquals(m.getMapping().get(String.class).start, 4);
    assertEquals(m.getMapping().get(String.class).length, 2);

  }
}