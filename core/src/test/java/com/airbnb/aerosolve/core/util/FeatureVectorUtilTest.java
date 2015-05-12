package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class FeatureVectorUtilTest {
  private static final Logger log = LoggerFactory.getLogger(FeatureVectorUtilTest.class);

  public FeatureVector makeFeatureVector(String key, double v1, double v2) {
    ArrayList list = new ArrayList<Double>();
    list.add(v1);
    list.add(v2);
    HashMap denseFeatures = new HashMap<String, List<Double>>();
    denseFeatures.put(key, list);
    FeatureVector featureVector = new FeatureVector();
    featureVector.setDenseFeatures(denseFeatures);
    return featureVector;
  }

  @Test
  public void testEmptyFeatureVectorMinKernel() {
    FeatureVector a = new FeatureVector();
    FeatureVector b = new FeatureVector();
    assertEquals(0.0, FeatureVectorUtil.featureVectorMinKernel(a, b), 0.1);
  }

  @Test
  public void testFeatureVectorMinKernelDifferentName() {
    FeatureVector a = makeFeatureVector("a", 1.0, 0.0);
    FeatureVector b = makeFeatureVector("b", 1.0, 0.0);
    assertEquals(0.0, FeatureVectorUtil.featureVectorMinKernel(a, b), 0.1);
  }

  @Test
  public void testFeatureVectorMinKernelNoOverlap() {
    FeatureVector a = makeFeatureVector("a", 1.0, 0.0);
    FeatureVector b = makeFeatureVector("a", 0.0, 1.0);
    assertEquals(0.0, FeatureVectorUtil.featureVectorMinKernel(a, b), 0.1);
  }

  @Test
  public void testFeatureVectorMinKernelSomeOverlap() {
    FeatureVector a = makeFeatureVector("a", 0.3, 0.7);
    FeatureVector b = makeFeatureVector("a", 0.0, 1.0);
    assertEquals(0.7, FeatureVectorUtil.featureVectorMinKernel(a, b), 0.1);
  }
}
