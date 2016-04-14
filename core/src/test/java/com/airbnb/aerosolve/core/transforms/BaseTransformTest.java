package com.airbnb.aerosolve.core.transforms;

import com.airbnb.aerosolve.core.features.DenseVector;
import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.FamilyVector;
import com.airbnb.aerosolve.core.features.Feature;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.features.FeatureValue;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.google.common.collect.ImmutableSet;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

/**
 *
 */
@Slf4j
public abstract class BaseTransformTest {

  protected final FeatureRegistry registry = new FeatureRegistry();

  abstract public String makeConfig();

  abstract public String configKey();

  // Hackz. This could be better.
  protected boolean runEmptyTest() {
    return true;
  }

  @Test
  public void testEmptyFeatureVector() {
    if (!runEmptyTest()) {
      return;
    }
    MultiFamilyVector featureVector = transformVector(
        TransformTestingHelper.makeEmptyVector(registry));

    assertTrue(featureVector.size() == 0);
  }

  protected MultiFamilyVector transformVector(MultiFamilyVector featureVector) {
    getTransform().apply(featureVector);
    return featureVector;
  }

  protected Transform<MultiFamilyVector> getTransform() {
    return getTransform(makeConfig(), configKey());
  }

  protected Transform<MultiFamilyVector> getTransform(String configStr, String configKey) {
    Config config = ConfigFactory.parseString(configStr);
    return TransformFactory.createTransform(config, configKey, registry, null);
  }

  public void assertStringFamily(MultiFamilyVector vector, String familyName,
                                 int expectedSize, Set<String> expected) {
    assertStringFamily(vector, familyName, expectedSize, expected, ImmutableSet.of());
  }

  // Set expectedSize to -1 if you don't want to test the size.
  public void assertStringFamily(MultiFamilyVector vector, String familyName,
                                 int expectedSize, Set<String> expected,
                                 Set<String> unexpectedKeys) {
    Family family = registry.family(familyName);
    FamilyVector fam = vector.get(family);
    assertNotNull(fam);
    for (FeatureValue value : fam) {
      log.info(value.toString());
    }
    assertTrue(fam.size() == expectedSize || expectedSize == -1);
    for (String name : expected) {
      assertTrue(fam.containsKey(family.feature(name)));
    }
    for (String name : unexpectedKeys) {
      assertFalse(fam.containsKey(family.feature(name)));
    }
  }

  protected void assertSparseFamily(MultiFamilyVector vector, String familyName,
                                    int expectedSize,
                                    Map<String, Double> expected) {
    assertSparseFamily(vector, familyName, expectedSize, expected, ImmutableSet.of());
  }

  // Set expectedSize to -1 if you don't want to test the size.
  protected void assertSparseFamily(MultiFamilyVector vector, String familyName,
                                    int expectedSize,
                                    Map<String, Double> expected,
                                    Set<String> unexpectedKeys) {
    Family family = registry.family(familyName);
    FamilyVector fam = vector.get(family);
    assertNotNull(fam);
    for (FeatureValue value : fam) {
      log.info(value.toString());
    }
    assertTrue(fam.size() == expectedSize || expectedSize == -1);
    for (Map.Entry<String, Double> entry : expected.entrySet()) {
      Feature feat = family.feature(entry.getKey());
      assertTrue(fam.containsKey(feat));
      assertEquals(fam.getDouble(feat), entry.getValue(), 0.01);
    }
    for (String name : unexpectedKeys) {
      assertFalse(fam.containsKey(family.feature(name)));
    }
  }

  public void assertDenseFamily(MultiFamilyVector vector, String familyName, double[] values) {
    FamilyVector fVec = vector.get(registry.family(familyName));
    assertTrue(fVec instanceof DenseVector);
    assertTrue(Arrays.equals(((DenseVector) fVec).denseArray(), values));
  }
}
