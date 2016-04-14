package com.airbnb.aerosolve.core.features;

import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class InputGenerationTest {
  private static final FeatureRegistry registry = new FeatureRegistry();
  public static MultiFamilyVector createVector() {
    Object[] values = new Object[6];
    String[] names = new String[6];

    names[0] = GenericNamingConvention.LABEL;
    values[0] = 5.0d;

    names[1] = "f_RAW";
    values[1] = "raw_feature";
    names[2] = "K_star";
    values[2] = "monkey";
    names[3] = "K_good";
    values[3] = false;
    names[4] = "S_speed";
    values[4] = 10.0d;

    names[5] = "X_jump";
    values[5] = null;

    BasicMultiFamilyVector vector = new BasicMultiFamilyVector(registry);
    return vector.putAll(names, values);
  }

  public static MultiFamilyVector createMulticlassVector() {
    Object[] values = new Object[1];
    String[] names = new String[1];

    names[0] = GenericNamingConvention.LABEL;
    values[0] = "a:1,b:2";

    BasicMultiFamilyVector vector = new BasicMultiFamilyVector(registry);
    return vector.putAll(names, values);
  }

  @Test
  public void addInputToGenerator() throws Exception {
    InputSchema m = getInputSchema();

    InputGenerator f = new InputGenerator(m);
    f.add(new double[]{Double.MIN_VALUE, 5}, Double.class);
    FeatureRegistry registry = new FeatureRegistry();
    BasicMultiFamilyVector vector = new BasicMultiFamilyVector(registry);
    f.load(vector);

    assertEquals(vector.getDouble(registry.feature("z", "b")), 5d, 0.1);
  }

  @Test
  public void addToSchema() throws Exception {
    InputSchema m = getInputSchema();

    assertEquals(m.getNames().length, 6);
    assertArrayEquals(m.getNames(),
                      new String[]{"z_a", "z_b", "z_c", "z_d", "z_e", "z_f"});
    assertEquals(m.getMapping().get(String.class).start, 4);
    assertEquals(m.getMapping().get(String.class).length, 2);
  }

  private InputSchema getInputSchema() {
    InputSchema m = new InputSchema(100);
    String[] doubleNames = {"z_a", "z_b"};
    m.add(Double.class, doubleNames);
    String[] booleanNames = {"z_c", "z_d"};
    m.add(Boolean.class, booleanNames);
    String[] strNames = {"z_e", "z_f"};
    m.add(String.class, strNames);
    m.finish();
    return m;
  }

  @Test
  public void testVector() throws Exception {
    MultiFamilyVector featureVector = createVector();

    // we don't want default BIAS families to be here.  We'll do that in the scorer.
    assertEquals(5, featureVector.numFamilies());
    FamilyVector fFamily = featureVector.get(registry.family("f"));
    assertEquals(1, fFamily.size());
    assertTrue(featureVector.containsKey(fFamily.family().feature("raw_feature")));

    FamilyVector kFamily = featureVector.get(registry.family("K"));
    assertEquals(2, kFamily.size());
    assertTrue(featureVector.containsKey(kFamily.family().feature("star:monkey")));
    assertTrue(featureVector.containsKey(kFamily.family().feature("good:F")));

    assertFalse(featureVector.contains(registry.family("X")));

    FamilyVector missFamily = featureVector.get(registry.family(GenericNamingConvention.MISS));
    assertEquals(1, missFamily.size());
    assertTrue(featureVector.containsKey(missFamily.family().feature("X_jump")));

    FamilyVector sFamily = featureVector.get(registry.family("S"));
    assertEquals(1, sFamily.size());
    assertEquals(featureVector.getDouble(sFamily.family().feature("speed")), 10d, .01);

    FamilyVector labelFamily = featureVector.get(registry.family(GenericNamingConvention.LABEL));
    assertEquals(1, labelFamily.size());
    assertEquals(featureVector.getDouble(
        labelFamily.family().feature(GenericNamingConvention.LABEL_FEATURE_NAME)), 5d, .01);
  }

  @Test
  public void testMulticlass() throws Exception {
    MultiFamilyVector featureVector = createMulticlassVector();

    assertEquals(1, featureVector.numFamilies());

    FamilyVector labelFamily = featureVector.get(registry.family(GenericNamingConvention.LABEL));
    assertEquals(2, labelFamily.size());
    assertEquals(1, labelFamily.get(labelFamily.family().feature("a")), 0);
    assertEquals(2, labelFamily.get(labelFamily.family().feature("b")), 0);
  }

  // TODO (Brad): Move tests to correct classes.

  /* @Test
  public void addNumberFeature() throws Exception {
    MultiFamilyVector vector = new FastMultiFamilyVector(registry);
    Feature feature = Features.calculateFeature("family_feature", 4d, registry);
    vector.put(feature, 4d);
    FamilyVector family = vector.get(registry.family("family"));
    assertEquals(1, family.size());
    assertEquals(4, family.get(family.family().feature("feature")), 0);

    feature = Features.calculateFeature("family_feature_float", 5d, registry);
    vector.put(feature, 5d);
    family = vector.get(registry.family("family"));
    assertEquals(2, family.size());
    assertEquals(5d, family.get(family.family().feature("feature_float")), 0);
  }

  @Test
  public void addBoolFeature() throws Exception {
    MultiFamilyVector vector = new FastMultiFamilyVector(registry);
    Feature feature = Features.calculateFeature("family_feature", false, registry);
    vector.putString(feature);
    FamilyVector family = vector.get(registry.family("family"));
    assertEquals(1, family.size());
    assertTrue(family.containsKey(family.family().feature("feature:F")));

    feature = Features.calculateFeature("family_feature", true, registry);
    vector.putString(feature);
    assertEquals(2, family.size());
    assertTrue(family.containsKey(family.family().feature("feature:T")));
    // Other one is still there.
    assertTrue(family.containsKey(family.family().feature("feature:F")));
  }

  @Test
  public void addStringFeature() throws Exception {
    MultiFamilyVector vector = new FastMultiFamilyVector(registry);
    Feature feature = Features.calculateFeature("family_feature", "value", registry);
    vector.putString(feature);

    feature = Features.calculateFeature("family_RAW", "feature_1", registry);
    vector.putString(feature);

    FamilyVector family = vector.get(registry.family("family"));
    assertEquals(2, family.size());
    assertTrue(family.containsKey(family.family().feature("feature:value")));
    assertTrue(family.containsKey(family.family().feature("feature_1")));
  }

  @Test
  public void addMultiClassLabel() throws Exception {
    MultiFamilyVector vector = empty();

    Family labelFamily = labelFamily();
    Features.addMultiClassLabel("a:1,b:2", vector, labelFamily);
    FamilyVector labelVector = vector.get(labelFamily);
    assertEquals(2, labelVector.size());
    assertEquals(1, labelVector.get(labelFamily.feature("a")), 0);
    assertEquals(2, labelVector.get(labelFamily.feature("b")), 0);
  }

  private MultiFamilyVector empty() {
    return new FastMultiFamilyVector(registry);
  }

  private Family labelFamily() {
    return registry.family(Features.LABEL);
  }

  @Test (expected = RuntimeException.class)
  public void addMultiClassManyColon() throws Exception {
    Features.addMultiClassLabel("a:1:2,b:2", empty(), labelFamily());
  }

  @Test (expected = RuntimeException.class)
  public void addMultiClassLabelNoolon() throws Exception {
    Features.addMultiClassLabel("abc,b:2", empty(), labelFamily());
  }

  @Test
  public void isLabel() throws Exception {
    assertTrue(Features.isLabel(Features.calculateFeature("LABEL", "A", registry), labelFamily()));
    assertFalse(Features.isLabel(Features.calculateFeature("LABE_ab", "A", registry), labelFamily()));
  }

  @Test(expected = RuntimeException.class)
  public void getFamilyEmpty() throws Exception {
    Features.calculateFeature("", 0d, registry);
  }

  @Test(expected = RuntimeException.class)
  public void getFamilyNotLABEL() throws Exception {
    Features.calculateFeature("LABE", 0d, registry);
  }

  @Test(expected = RuntimeException.class)
  public void getFamilyPrefix() throws Exception {
    Features.calculateFeature("_abc", 0d, registry);
  }

  @Test
  public void getFamily() throws Exception {
    Feature feature = Features.calculateFeature("f_ab_cd", 0d, registry);
    assertEquals(feature.family(), registry.family("f"));
    assertEquals(feature.name(), "ab_cd");
  } */
}