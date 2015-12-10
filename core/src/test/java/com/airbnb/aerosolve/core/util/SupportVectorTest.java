package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.Example;

import com.airbnb.aerosolve.core.FeatureVector;
import com.airbnb.aerosolve.core.FunctionForm;
import com.airbnb.aerosolve.core.ModelRecord;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Set;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class SupportVectorTest {
  private static final Logger log = LoggerFactory.getLogger(SupportVectorTest.class);

  @Test
  public void testRbf() {
    FloatVector v1 = new FloatVector(new float[]{1.0f, 2.0f});
    FloatVector v2 = new FloatVector(new float[]{3.0f, 5.0f});

    SupportVector sv = new SupportVector(v1, FunctionForm.RADIAL_BASIS_FUNCTION, 0.1f, 0.5f);

    assertEquals(0.5, sv.evaluate(v1), 0.01f);
    assertEquals(0.5 * Math.exp(- 0.1 * (4.0 + 9.0)), sv.evaluate(v2), 0.01f);
  }

  @Test
  public void testArcCosine() {
    FloatVector v1 = new FloatVector(new float[]{1.0f, 2.0f});
    FloatVector v2 = new FloatVector(new float[]{3.0f, 5.0f});

    SupportVector sv = new SupportVector(v1, FunctionForm.ARC_COSINE, 0.1f, 0.5f);

    assertEquals(0.5, sv.evaluate(v1), 0.01f);
    double expected = 1.0 - (float) Math.acos( (3.0 + 10.0) / Math.sqrt((1.0 + 4.0) * (9.0 + 25.0))  ) / Math.PI;
    assertEquals(0.5* expected, sv.evaluate(v2), 0.01f);
  }

  @Test
  public void testSerialization() {
    FloatVector v1 = new FloatVector(new float[]{1.0f, 2.0f});
    FloatVector v2 = new FloatVector(new float[]{3.0f, 5.0f});

    SupportVector tmp = new SupportVector(v1, FunctionForm.ARC_COSINE, 0.1f, 0.5f);

    ModelRecord rec = tmp.toModelRecord();
    SupportVector sv = new SupportVector(rec);

    assertEquals(0.5, sv.evaluate(v1), 0.01f);
    double expected = 1.0 - (float) Math.acos( (3.0 + 10.0) / Math.sqrt((1.0 + 4.0) * (9.0 + 25.0))  ) / Math.PI;
    assertEquals(0.5 * expected, sv.evaluate(v2), 0.01f);
  }

}