package com.airbnb.aerosolve.core.features;

import com.airbnb.aerosolve.core.features.FeatureGen;
import com.airbnb.aerosolve.core.features.FeatureMapping;
import com.airbnb.aerosolve.core.features.Features;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class FeatureGenTest {

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

    FeatureGen f = new FeatureGen(m);
    f.add(new float[]{Float.MIN_VALUE, 5}, Double.class);
    Features p = f.gen();
    assertEquals(p.names.length, 6);
    assertEquals(p.values[0], null);
    assertEquals((Double) p.values[1], 5, 0.1);

  }
}