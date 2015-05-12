package com.airbnb.aerosolve.core.util;

import com.airbnb.aerosolve.core.Example;
import com.airbnb.aerosolve.core.FeatureVector;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Hector Yee
 */
public class FeatureDictionaryTest {
  private static final Logger log = LoggerFactory.getLogger(FeatureDictionaryTest.class);

  public FeatureVector makeDenseFeatureVector(String id,
                                              double v1,
                                              double v2) {
    ArrayList list = new ArrayList<Double>();
    list.add(v1);
    list.add(v2);
    HashMap denseFeatures = new HashMap<String, List<Double>>();
    denseFeatures.put("a", list);
    ArrayList list2 = new ArrayList<Double>();
    list2.add(v2);
    list2.add(v1);
    denseFeatures.put("b", list2);
    Set<String> list3 = new HashSet<>();
    list3.add(id);
    Map<String, Set<String>> stringFeatures = new HashMap<>();
    stringFeatures.put("id", list3);
    FeatureVector featureVector = new FeatureVector();
    featureVector.setDenseFeatures(denseFeatures);
    featureVector.setStringFeatures(stringFeatures);
    return featureVector;
  }

  public FeatureVector makeSparseFeatureVector(String id,
                                               String v1,
                                               String v2) {
    Set<String> set = new HashSet<String>();
    set.add(v1);
    set.add(v2);
    Map<String, Set<String>> stringFeatures = new HashMap<>();
    stringFeatures.put("a", set);
    Set<String> set2 = new HashSet<>();
    set2.add(id);
    stringFeatures.put("id", set2);
    FeatureVector featureVector = new FeatureVector();
    featureVector.setStringFeatures(stringFeatures);
    return featureVector;
  }

  @Test
  public void testDictionaryMinKernel() {
    List<FeatureVector> dict = new ArrayList<>();
    dict.add(makeDenseFeatureVector("0", 0.3, 0.7));
    dict.add(makeDenseFeatureVector("1", 0.7, 0.3));
    dict.add(makeDenseFeatureVector("2", 0.5, 0.5));
    dict.add(makeDenseFeatureVector("3", 0.0, 0.0));
    FeatureDictionary dictionary = new MinKernelDenseFeatureDictionary();
    dictionary.setDictionaryList(dict);

    KNearestNeighborsOptions opt = new KNearestNeighborsOptions();
    opt.setIdKey("id");
    opt.setOutputKey("mk");
    opt.setNumNearest(2);

    FeatureVector response1 = dictionary.getKNearestNeighbors(opt, dict.get(0));
    log.info(response1.toString());
    assertTrue(response1.floatFeatures != null);
    assertEquals(response1.getFloatFeatures().get("mk").size(), 2);
    assertEquals(response1.getFloatFeatures().get("mk").get("1"), 1.2, 0.1);
    assertEquals(response1.getFloatFeatures().get("mk").get("2"), 1.6, 0.1);

    FeatureVector response2 = dictionary.getKNearestNeighbors(opt, dict.get(1));
    log.info(response2.toString());
    assertTrue(response2.floatFeatures != null);
    assertEquals(response2.getFloatFeatures().get("mk").size(), 2);
    assertEquals(response2.getFloatFeatures().get("mk").get("0"), 1.2, 0.1);
    assertEquals(response2.getFloatFeatures().get("mk").get("2"), 1.6, 0.1);

    FeatureVector response3 = dictionary.getKNearestNeighbors(opt, dict.get(2));
    log.info(response3.toString());
    assertTrue(response3.floatFeatures != null);
    assertEquals(response3.getFloatFeatures().get("mk").size(), 2);
    assertEquals(response3.getFloatFeatures().get("mk").get("0"), 1.6, 0.1);
    assertEquals(response3.getFloatFeatures().get("mk").get("1"), 1.6, 0.1);
  }

  @Test
  public void testDictionaryLSH() {
    List<FeatureVector> dict = new ArrayList<>();
    dict.add(makeSparseFeatureVector("0", "a", "b"));
    dict.add(makeSparseFeatureVector("1", "b", "c"));
    dict.add(makeSparseFeatureVector("2", "a", "e"));
    dict.add(makeSparseFeatureVector("3", "e", "f"));
    dict.add(makeSparseFeatureVector("4", "@@@", "$$$"));
    dict.add(makeSparseFeatureVector("5", "$$$", "@@@"));
    FeatureDictionary dictionary = new LocalitySensitiveHashSparseFeatureDictionary();
    dictionary.setDictionaryList(dict);

    KNearestNeighborsOptions opt = new KNearestNeighborsOptions();
    opt.setIdKey("id");
    opt.setOutputKey("sim");
    opt.setNumNearest(2);
    opt.setFeatureKey("a");

    FeatureVector response1 = dictionary.getKNearestNeighbors(opt, dict.get(0));
    log.info(response1.toString());
    assertTrue(response1.floatFeatures != null);
    assertEquals(response1.getFloatFeatures().get("sim").size(), 2);
    assertEquals(response1.getFloatFeatures().get("sim").get("1"), 1.0, 0.1);
    assertEquals(response1.getFloatFeatures().get("sim").get("2"), 1.0, 0.1);

    FeatureVector response2 = dictionary.getKNearestNeighbors(opt, dict.get(1));
    log.info(response2.toString());
    assertTrue(response2.floatFeatures != null);
    assertEquals(response2.getFloatFeatures().get("sim").size(), 1);
    assertEquals(response2.getFloatFeatures().get("sim").get("0"), 1.0, 0.1);

    FeatureVector response3 = dictionary.getKNearestNeighbors(opt, dict.get(2));
    log.info(response3.toString());
    assertTrue(response3.floatFeatures != null);
    assertEquals(response3.getFloatFeatures().get("sim").size(), 2);
    assertEquals(response3.getFloatFeatures().get("sim").get("0"), 1.0, 0.1);
    assertEquals(response3.getFloatFeatures().get("sim").get("3"), 1.0, 0.1);

    FeatureVector response4 = dictionary.getKNearestNeighbors(opt, dict.get(3));
    log.info(response4.toString());
    assertTrue(response4.floatFeatures != null);
    assertEquals(response4.getFloatFeatures().get("sim").size(), 1);
    assertEquals(response4.getFloatFeatures().get("sim").get("2"), 1.0, 0.1);

    FeatureVector response5 = dictionary.getKNearestNeighbors(opt, dict.get(4));
    log.info(response5.toString());
    assertTrue(response5.floatFeatures != null);
    assertEquals(response5.getFloatFeatures().get("sim").size(), 1);
    assertEquals(response5.getFloatFeatures().get("sim").get("5"), 2.0, 0.1);
  }
}
