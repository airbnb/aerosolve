package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.DebugScoreDiffRecord;
import com.airbnb.aerosolve.core.DebugScoreRecord;
import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class UtilTest {
  private static final Logger log = LoggerFactory.getLogger(UtilTest.class);

  public FeatureVector makeFeatureVector() {
    Set list = new HashSet<String>();
    list.add("aaa");
    list.add("bbb");
    HashMap stringFeatures = new HashMap<String, ArrayList<String>>();
    stringFeatures.put("string_feature", list);
    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    return featureVector;
  }

  @Test
  public void testEncodeDecodeFeatureVector() {
    FeatureVector featureVector = makeFeatureVector();
    String str = Util.encode(featureVector);
    assertTrue(str.length() > 0);
    log.info(str);
    FeatureVector featureVector2 = Util.decodeFeatureVector(str);
    assertTrue(featureVector2.stringFeatures != null);
    assertTrue(featureVector2.stringFeatures.containsKey("string_feature"));
    Set<String> list2 = featureVector2.stringFeatures.get("string_feature");
    assertTrue(list2.size() == 2);
  }

  @Test
  public void testEncodeDecodeExample() {
    FeatureVector featureVector = makeFeatureVector();
    Example example = new Example();
    example.addToExample(featureVector);
    String str = Util.encode(example);
    assertTrue(str.length() > 0);
    log.info(str);
    Example example2 = Util.decodeExample(str);
    assertTrue(example2.example.size() == 1);
  }

  @Test
  public void testFlattenFeature() {
    FeatureVector featureVector = makeFeatureVector();
    Map<String, Map<String, Double>> floatFeatures = new HashMap<>();
    Map<String, Double> tmp = new HashMap<String, Double>();
    floatFeatures.put("float_feature", tmp);
    tmp.put("x", 0.3);
    tmp.put("y", -0.2);
    featureVector.floatFeatures = floatFeatures;
    Map<String, Map<String, Double>> flatFeature = Util.flattenFeature(featureVector);
    assertEquals(1.0, flatFeature.get("string_feature").get("aaa"), 0.1);
    assertEquals(1.0, flatFeature.get("string_feature").get("bbb"), 0.1);
    assertEquals(0.3, flatFeature.get("float_feature").get("x"), 0.1);
    assertEquals(-0.2, flatFeature.get("float_feature").get("y"), 0.1);
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

  @Test
  public void testGetIntersection() {
    Set<Integer> a = new HashSet<>(Arrays.asList(1, 2));
    Set<Integer> b = new HashSet<>(Arrays.asList(3, 4));

    Set<Integer> r = Util.getIntersection(a, b);
    assertEquals(0, r.size());

    a = new HashSet<>(Arrays.asList(1, 2));
    b = new HashSet<>(Arrays.asList(2, 4));
    r = Util.getIntersection(a, b);
    assertEquals(1, r.size());
    assertTrue(r.contains(2));

    a = new HashSet<>(Arrays.asList(1, 2, 3));
    b = new HashSet<>(Arrays.asList(1, 2, 3));
    r = Util.getIntersection(a, b);
    assertEquals(3, r.size());
    assertEquals(a, r);

    r = Util.getIntersection(null, b);
    assertEquals(0, r.size());
  }
}
