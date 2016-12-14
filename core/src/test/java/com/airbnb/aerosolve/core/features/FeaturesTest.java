package com.airbnb.aerosolve.core.features;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.junit.Test;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static org.junit.Assert.*;

public class FeaturesTest {
  public static Features createFeature() {
    Object[] values = new Object[7];
    String[] names = new String[7];

    names[0] = Features.LABEL;
    values[0] = new Double(5.0);

    names[1] = "f_RAW";
    values[1] = "raw_feature";
    names[2] = "K_star";
    values[2] = "monkey";
    names[3] = "K_good";
    values[3] = Boolean.FALSE;
    names[4] = "S_speed";
    values[4] = new Double(10.0);

    names[5] = "X_jump";
    values[5] = null;

    names[6] = "_meta_id_listing";
    values[6] = 12345;
    return Features.builder().names(names).values(values).build();
  }

  public static Features createMultiClassFeature() {
    Object[] values = new Object[1];
    String[] names = new String[1];

    names[0] = Features.LABEL;
    values[0] = "a:1,b:2";
    return Features.builder().names(names).values(values).build();
  }

  @Test
  public void toExample() throws Exception {
    Example example = createFeature().toExample(false);
    FeatureVector featureVector = example.getExample().get(0);
    final Map<String, Set<String>> stringFeatures = featureVector.getStringFeatures();
    final Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();
    final Map<String, String> metadata = example.getMetadata();

    // we have default BIAS
    assertEquals(4, stringFeatures.size());
    Set<String> stringFeature = stringFeatures.get("f");
    assertEquals(1, stringFeature.size());
    assertTrue(stringFeature.contains("raw_feature"));

    stringFeature = stringFeatures.get("K");
    assertEquals(2, stringFeature.size());
    assertTrue(stringFeature.contains("star:monkey"));
    assertTrue(stringFeature.contains("good:F"));

    stringFeature = stringFeatures.get("X");
    assertNull(stringFeature);

    stringFeature = stringFeatures.get(Features.MISS);
    assertEquals(1, stringFeature.size());
    assertTrue(stringFeature.contains("X_jump"));

    assertEquals(2, floatFeatures.size());
    Map<String, Double> floatFeature = floatFeatures.get("S");
    assertEquals(1, floatFeature.size());
    assertEquals(10.0, floatFeature.get("speed"), 0);

    floatFeature = floatFeatures.get(Features.LABEL);
    assertEquals(1, floatFeature.size());
    assertEquals(5.0, floatFeature.get(Features.LABEL_FEATURE_NAME), 0);

    assertEquals(1, metadata.size());
    assertEquals("12345", metadata.get("id_listing"));
  }

  @Test
  public void toExampleMultiClass() throws Exception {
    Example example = createMultiClassFeature().toExample(true);
    FeatureVector featureVector = example.getExample().get(0);
    final Map<String, Map<String, Double>> floatFeatures = featureVector.getFloatFeatures();

    assertEquals(1, floatFeatures.size());
    Map<String, Double> floatFeature = floatFeatures.get(Features.LABEL);
    assertEquals(2, floatFeature.size());
    assertEquals(1, floatFeature.get("a"), 0);
    assertEquals(2, floatFeature.get("b"), 0);
  }

  @Test
  public void addNumberFeature() throws Exception {
    Pair<String, String> featurePair = new ImmutablePair<>("family", "feature");
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();
    Features.addNumberFeature(4, featurePair, floatFeatures);
    Map<String, Double> feature = floatFeatures.get("family");
    assertEquals(1, feature.size());
    assertEquals(4, feature.get("feature"), 0);

    featurePair = new ImmutablePair<>("family", "feature_float");
    Features.addNumberFeature(5.0f, featurePair, floatFeatures);
    assertEquals(2, feature.size());
    assertEquals(5.0, feature.get("feature_float"), 0);
  }

  @Test
  public void addBoolFeature() throws Exception {
    Pair<String, String> featurePair = new ImmutablePair<>("family", "feature");
    Map<String, Set<String>> stringFeatures = new HashMap<>();
    Features.addBoolFeature(false, featurePair, stringFeatures);
    Features.addBoolFeature(true, featurePair, stringFeatures);
    Set<String> feature = stringFeatures.get("family");
    assertEquals(2, feature.size());
    assertTrue(feature.contains("feature:T"));
    assertTrue(feature.contains("feature:F"));
  }

  @Test
  public void addStringFeature() throws Exception {
    Pair<String, String> featurePair = new ImmutablePair<>("family", "feature");
    Map<String, Set<String>> stringFeatures = new HashMap<>();
    Features.addStringFeature("value", featurePair, stringFeatures);

    Pair<String, String> raw = new ImmutablePair<>("family", Features.RAW);
    Features.addStringFeature("feature_1", raw, stringFeatures);

    Set<String> feature = stringFeatures.get("family");
    assertEquals(2, feature.size());
    assertTrue(feature.contains("feature:value"));
    assertTrue(feature.contains("feature_1"));
  }

  @Test
  public void addMultiClassLabel() throws Exception {
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();

    Features.addMultiClassLabel("a:1,b:2", floatFeatures);
    Map<String, Double> feature = floatFeatures.get(Features.LABEL);
    assertEquals(2, feature.size());
    assertEquals(1, feature.get("a"), 0);
    assertEquals(2, feature.get("b"), 0);
  }

  @Test (expected = RuntimeException.class)
  public void addMultiClassManyColon() throws Exception {
    Features.addMultiClassLabel("a:1:2,b:2", Collections.EMPTY_MAP);
  }

  @Test (expected = RuntimeException.class)
  public void addMultiClassLabelNoolon() throws Exception {
    Features.addMultiClassLabel("abc,b:2", Collections.EMPTY_MAP);
  }

  @Test
  public void isLabel() throws Exception {
    assertTrue(Features.isLabel(Features.getFamily("LABEL")));
    assertFalse(Features.isLabel(Features.getFamily("LABE_ab")));
  }

  @Test(expected = RuntimeException.class)
  public void getFamilyEmpty() throws Exception {
    Pair<String, String> p = Features.getFamily("");
  }

  @Test
  public void getDefaultStringFamily() throws Exception {
    Pair<String, String> p = Features.getFamily("string");
    assertEquals("", p.getLeft());
    assertEquals(Features.DEFAULT_STRING_FAMILY, Features.getStringFamily(p));
  }

  @Test
  public void getDefaultFloatFamily() throws Exception {
    Pair<String, String> p = Features.getFamily("float");
    assertEquals("", p.getLeft());
    assertEquals(Features.DEFAULT_FLOAT_FAMILY, Features.getFloatFamily(p));
  }

  @Test(expected = RuntimeException.class)
  public void getFamilyPrefix() throws Exception {
    Pair<String, String> p = Features.getFamily("_abc");
  }

  @Test
  public void getFamily() throws Exception {
    Pair<String, String> p = Features.getFamily("f_ab_cd");
    assertEquals(p.getLeft(), "f");
    assertEquals(p.getRight(), "ab_cd");
  }
}
