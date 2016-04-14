package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.DebugScoreDiffRecord;
import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.features.Family;
import com.airbnb.aerosolve.core.features.FeatureRegistry;
import com.airbnb.aerosolve.core.features.MultiFamilyVector;
import com.airbnb.aerosolve.core.features.SimpleExample;
import com.airbnb.aerosolve.core.transforms.TransformTestingHelper;
import com.google.common.collect.Sets;
import java.util.ArrayList;
import java.util.List;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
@Slf4j
public class UtilTest {
  private final FeatureRegistry registry = new FeatureRegistry();

  @Test
  public void testEncodeDecodeFeatureVector() {
    MultiFamilyVector featureVector = makeFeatureVector();
    String str = Util.encodeFeatureVector(featureVector);
    assertTrue(str.length() > 0);
    log.info(str);
    MultiFamilyVector featureVector2 = Util.decodeFeatureVector(str, registry);
    assertTrue(featureVector2.numFamilies() == 2);
    Family stringFamily = registry.family("string_feature");
    assertTrue(featureVector2.contains(stringFamily));
    assertTrue(featureVector2.get(stringFamily).size() == 2);

    Family sparseFamily = registry.family("sparse_feature");
    assertTrue(featureVector2.contains(sparseFamily));
    assertTrue(featureVector2.get(sparseFamily).size() == 2);
  }

  private MultiFamilyVector makeFeatureVector() {
    MultiFamilyVector vector = TransformTestingHelper.makeEmptyVector(registry);

    Family stringFamily = registry.family("string_feature");
    vector.putString(stringFamily.feature("aaa"));
    vector.putString(stringFamily.feature("bbb"));

    Family sparseFamily = registry.family("sparse_feature");

    vector.put(sparseFamily.feature("lat"), 37.7);
    vector.put(sparseFamily.feature("long"), 40.0);

    return vector;
  }

  @Test
  public void testEncodeDecodeExample() {
    MultiFamilyVector featureVector = makeFeatureVector();
    Example example = new SimpleExample(registry);
    example.addToExample(featureVector);
    String str = Util.encodeExample(example);
    assertTrue(str.length() > 0);
    log.info(str);
    Example example2 = Util.decodeExample(str, registry);
    assertTrue(Sets.newHashSet(example2).size() == 1);
  }

  @Test
  public void testCompareDebugRecords(){
    List<DebugScoreRecord> record1 = new ArrayList<>();
    List<DebugScoreRecord> record2 = new ArrayList<>();

    DebugScoreRecord r11 = new DebugScoreRecord();
    r11.setFeatureFamily("a_family");
    r11.setFeatureName("a_name");
    r11.setFeatureValue(0.0);
    r11.setFeatureWeight(0.0);

    DebugScoreRecord r12 = new DebugScoreRecord();
    r12.setFeatureFamily("b_family");
    r12.setFeatureName("b_name");
    r12.setFeatureValue(0.1);
    r12.setFeatureWeight(1.0);

    DebugScoreRecord r13 = new DebugScoreRecord();
    r13.setFeatureFamily("c_family");
    r13.setFeatureName("c_name");
    r13.setFeatureValue(0.2);
    r13.setFeatureWeight(1.0);

    DebugScoreRecord r21 = new DebugScoreRecord();
    r21.setFeatureFamily("a_family");
    r21.setFeatureName("a_name");
    r21.setFeatureValue(0.0);
    r21.setFeatureWeight(1.0); // weight_diff = 0.0 - 1.0 = -1.0

    DebugScoreRecord r22 = new DebugScoreRecord();
    r22.setFeatureFamily("b_family");
    r22.setFeatureName("b_name");
    r22.setFeatureValue(0.1);
    r22.setFeatureWeight(0.5); // weight_diff = 1.0 - 0.5 = 0.5

    DebugScoreRecord r23 = new DebugScoreRecord();
    r23.setFeatureFamily("c_family");
    r23.setFeatureName("c_name");
    r23.setFeatureValue(0.2);
    r23.setFeatureWeight(-1.0); // 1.0 - (-1.0) = 2.0

    record1.add(r11);
    record2.add(r21);
    List<DebugScoreDiffRecord> result1 = Util.compareDebugRecords(record1, record2);
    assertEquals(1, result1.size());
    assertEquals(-1.0, result1.get(0).getFeatureWeightDiff(), 0.0001);

    record2.add(r22); // record2 has one more record
    List<DebugScoreDiffRecord> result2 = Util.compareDebugRecords(record1, record2);
    assertEquals(2, result2.size());
    assertEquals(-1.0, result2.get(0).getFeatureWeightDiff(), 0.0001);
    assertEquals(-0.5, result2.get(1).getFeatureWeightDiff(), 0.0001);

    record1.add(r12);
    record1.add(r13);
    record2.add(r23);
    List<DebugScoreDiffRecord> result3 = Util.compareDebugRecords(record1, record2);
    assertEquals(3, result3.size());
    assertEquals(2.0, result3.get(0).getFeatureWeightDiff(), 0.0001);
    assertEquals(-1.0, result3.get(1).getFeatureWeightDiff(), 0.0001);
    assertEquals(0.5, result3.get(2).getFeatureWeightDiff(), 0.0001);
  }
}